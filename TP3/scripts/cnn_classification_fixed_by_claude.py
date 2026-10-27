import argparse
from collections import Counter
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class EmbModel(Enum):
    FastText = 0
    W2V_CBow = 1
    W2V_Skipgram = 2


sequence_length = 128

# Évaluation seqeval : tags attendus au format BIO (B-PER, I-PER, O).
# Si tes données sont en BIOES/BILOU, change SCHEME (ex: from seqeval.scheme import BILOU).
SCHEME = IOB2
MODE = 'strict'

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=['cnn', 'lstm'], required=True, help="Type de modèle : cnn ou lstm.")
parser.add_argument("--emb-model", type=int, choices=[0, 1, 2], required=True,
                    help="0: FastText, 1: Word2Vec CBow, 2: Word2Vec Skipgram")
parser.add_argument("--emb-model-path", type=str, required=True, help="Chemin du modèle d'embeddings")
parser.add_argument("--train", type=str, required=True, help="Données d'entraînement au format CoNLL")
parser.add_argument("--valid", type=str, required=True, help="Données de validation (dev) au format CoNLL")
parser.add_argument("--test", type=str, required=True, help="Données d'évaluation au format CoNLL")
parser.add_argument("--epochs", default=1, type=int, help="Nombre d'epochs")
parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
parser.add_argument("--max-vocab", default=-1, type=int,
                    help="Taille max du vocabulaire (-1 = illimité, recommandé en NER)")

args = parser.parse_args()

train_file = args.train
valid_file = args.valid   # (corrigé : valid/test étaient inversés)
test_file = args.test
mymodel = args.model
emb_model = EmbModel(args.emb_model)
emb_model_path = args.emb_model_path
lr = args.lr
epochs = args.epochs
max_vocab = args.max_vocab

print(emb_model)


# --------------------------------------------------------------------------- #
# Lecture des données
# --------------------------------------------------------------------------- #
def read_conll(path):
    sentences, labels = [], []
    tokens, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":              # ligne vide = fin de phrase
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
                continue
            if line.startswith("#"):            # ligne de commentaire
                continue
            parts = line.split()
            if len(parts) < 5:                  # ligne malformée
                continue
            tokens.append(parts[1])             # token
            tags.append(parts[4])               # tag
    if tokens:                                  # dernière phrase sans ligne vide finale
        sentences.append(tokens)
        labels.append(tags)
    return sentences, labels


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def load_embeddings(emb_model, path):
    if emb_model == EmbModel.FastText:
        return load_facebook_vectors(path)
    if emb_model in (EmbModel.W2V_CBow, EmbModel.W2V_Skipgram):
        return KeyedVectors.load_word2vec_format(path, binary=False)
    raise ValueError("Unknown embedder.")


def build_embedding_matrix(kv, word2int):
    """Matrice d'embeddings alignée sur word2int : la ligne i = vecteur du mot d'indice i."""
    unk_vec = kv.vectors.mean(axis=0)
    matrix = np.zeros((len(word2int), kv.vector_size), dtype=np.float32)  # ligne 0 = PAD = zéros
    n_oov = 0
    for word, idx in word2int.items():
        if word == '<PAD>':
            continue
        if word == '<UNK>':
            matrix[idx] = unk_vec
            continue
        try:
            matrix[idx] = kv[word]  # FastText : fonctionne aussi hors vocabulaire (n-grammes)
        except KeyError:
            matrix[idx] = unk_vec
            n_oov += 1
    print(f"Embedding matrix: {matrix.shape}, mots sans vecteur (-> UNK) : {n_oov}")
    return torch.from_numpy(matrix)


# --------------------------------------------------------------------------- #
# Modèles
# --------------------------------------------------------------------------- #
class NERModelLSTM(nn.Module):
    def __init__(self, embedding_weights, output_size, hidden_size=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.name = "lstm"
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=True, padding_idx=0)
        embedding_size = embedding_weights.shape[1]

        # Bidirectionnel : regarder le mot suivant aide à classer le token courant.
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers,
                            dropout=dropout if n_layers > 1 else 0.0,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = self.embedding(x.long())
        o, _ = self.lstm(x)
        o = self.dropout(o)
        return self.fc(o)                       # (N, W, C)


class NERModelCNN(nn.Module):
    def __init__(self, embedding_weights, class_size, dropout=0.2):
        super().__init__()
        self.name = "cnn"

        Ci, Co, Ks = 1, 100, [3, 5, 7]          # Ks impairs : padding K//2 conserve la longueur
        self.embed = nn.Embedding.from_pretrained(embedding_weights, freeze=True, padding_idx=0)
        embedding_size = embedding_weights.shape[1]

        self.convs = nn.ModuleList(
            [nn.Conv2d(Ci, Co, (K, embedding_size), padding=(K // 2, 0)) for K in Ks]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, class_size)

    def forward(self, x):
        x = self.embed(x.long())                # (N, W, D)
        x = x.unsqueeze(1)                      # (N, 1, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]   # [(N, Co, W), ...]
        x = [i.permute(0, 2, 1) for i in x]     # [(N, W, Co), ...]
        x = torch.cat(x, dim=2)                 # (N, W, len(Ks)*Co)
        x = self.dropout(x)
        return self.fc1(x)                      # (N, W, C)


# --------------------------------------------------------------------------- #
# Prétraitement
# --------------------------------------------------------------------------- #
def pad_features(sequences, pad_id, seq_length=128):
    features = np.full((len(sequences), seq_length), pad_id, dtype=int)
    for i, row in enumerate(sequences):
        row = np.array(row)[:seq_length]        # les séquences trop longues sont tronquées
        features[i, :len(row)] = row
    return features


def encode_sequences(sequences, index, pad_id, seq_length=128, is_target=False):
    """Encode une liste de phrases (ou de listes de tags) en indices entiers."""
    encoded = []
    for seq in tqdm(sequences):
        l_enc = []
        for item in seq:
            if item in index:
                l_enc.append(index[item])
            else:
                l_enc.append(index.get('<UNK>', 0) if not is_target else 0)
        encoded.append(l_enc)

    x = pad_features(encoded, pad_id=pad_id, seq_length=seq_length)
    assert len(x) == len(sequences)
    assert len(x[0]) == seq_length
    return x


def load_and_preprocess_data(filename_train, filename_valid, filename_test, seq_length=128, max_vocab=-1):
    print("Loading CoNLL files...")
    train_sentences, train_labels = read_conll(filename_train)
    valid_sentences, valid_labels = read_conll(filename_valid)
    test_sentences, test_labels = read_conll(filename_test)

    for name, sents in [('train', train_sentences), ('valid', valid_sentences), ('test', test_sentences)]:
        n_long = sum(len(s) > seq_length for s in sents)
        if n_long:
            print(f"[WARNING] {name}: {n_long}/{len(sents)} phrases > {seq_length} tokens (tronquées)")

    print("Building vocabulary and tag dictionaries...")

    # Vocabulaire sur tous les splits : sans risque ici, les embeddings sont pré-entraînés et gelés.
    all_words = [w for sent in train_sentences + valid_sentences + test_sentences for w in sent]
    counter = Counter(all_words)
    vocab = sorted(counter, key=counter.get, reverse=True)
    if max_vocab != -1:
        vocab = vocab[:max_vocab]

    int2word = dict(enumerate(vocab, 2))
    int2word[0] = '<PAD>'
    int2word[1] = '<UNK>'
    word2int = {word: i for i, word in int2word.items()}

    # Tags NER (B-PER, I-PER, O...) ; l'indice 0 est réservé au padding.
    all_tags = set(tag for tags in train_labels + valid_labels + test_labels for tag in tags)
    tag2int = {tag: i + 1 for i, tag in enumerate(sorted(all_tags))}
    tag2int['<PAD>'] = 0

    print("Encoding sequences...")
    pad_w, pad_t = word2int['<PAD>'], tag2int['<PAD>']
    l_train_x = encode_sequences(train_sentences, word2int, pad_w, seq_length)
    l_valid_x = encode_sequences(valid_sentences, word2int, pad_w, seq_length)
    l_test_x = encode_sequences(test_sentences, word2int, pad_w, seq_length)

    l_train_y = encode_sequences(train_labels, tag2int, pad_t, seq_length, is_target=True)
    l_valid_y = encode_sequences(valid_labels, tag2int, pad_t, seq_length, is_target=True)
    l_test_y = encode_sequences(test_labels, tag2int, pad_t, seq_length, is_target=True)

    return l_train_x, l_train_y, l_valid_x, l_valid_y, l_test_x, l_test_y, word2int, tag2int


train_x, train_y, valid_x, valid_y, test_x, test_y, word2int, tag2int = load_and_preprocess_data(
    train_file, valid_file, test_file, sequence_length, max_vocab)
int2tag = {i: t for t, i in tag2int.items()}

print('Feature Shapes:')
print('===============')
print('Train set: {}'.format(train_x.shape))
print('Validation set: {}'.format(valid_x.shape))
print('Test set: {}'.format(test_x.shape))

batch_size = 128

print("Create Tensor dataset")
trainset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
validset = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
testset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
valloader = DataLoader(validset, shuffle=False, batch_size=batch_size)
testloader = DataLoader(testset, shuffle=False, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hidden_size = 128
n_layers = 1
dropout = 0.25
num_classes = len(tag2int)

# --------------------------------------------------------------------------- #
# Modèle
# --------------------------------------------------------------------------- #
print("Load embeddings")
kv = load_embeddings(emb_model, emb_model_path)
embedding_weights = build_embedding_matrix(kv, word2int)   # alignée sur word2int
del kv

print("Initialize model")
if mymodel == 'lstm':
    model = NERModelLSTM(embedding_weights, output_size=num_classes,
                         hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
else:
    model = NERModelCNN(embedding_weights, num_classes)
print(model)

# --------------------------------------------------------------------------- #
# Loss avec poids de classes lissés (hors padding)
# --------------------------------------------------------------------------- #
counts = np.bincount(train_y[train_y != 0], minlength=num_classes).astype(float)
present = counts > 0
w = np.ones(num_classes)
w[present] = np.sqrt(counts[present].sum() / (present.sum() * counts[present]))  # 'balanced' lissé (sqrt)
w[present] /= w[present].mean()
weights_tensor = torch.FloatTensor(w).to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=0)
optim = Adam(model.parameters(), lr=lr)
grad_clip = 5


# --------------------------------------------------------------------------- #
# Utilitaires d'évaluation (seqeval)
# --------------------------------------------------------------------------- #
def predict(out):
    """argmax en interdisant la classe <PAD> (indice 0)."""
    out = out.detach().clone()
    out[..., 0] = float('-inf')
    return out.argmax(dim=-1)


def decode(predicted, target):
    """Tenseurs (N, L) -> deux listes de listes de tags (une liste par phrase), sans padding."""
    y_true, y_pred = [], []
    for p_row, t_row in zip(predicted.cpu().numpy(), target.cpu().numpy()):
        keep = t_row != 0
        y_true.append([int2tag[i] for i in t_row[keep]])
        y_pred.append([int2tag[i] for i in p_row[keep]])
    return y_true, y_pred


def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for feature, target in loader:
            feature, target = feature.to(device), target.to(device)
            out = model(feature)
            total_loss += criterion(out.reshape(-1, num_classes), target.reshape(-1)).item()

            predicted = predict(out)
            mask = target != 0
            correct += ((predicted == target) & mask).sum().item()
            total += mask.sum().item()

            t, p = decode(predicted, target)
            y_true += t
            y_pred += p
    return total_loss / len(loader), correct / max(total, 1), y_true, y_pred


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'epochs': epochs}
es_limit = 5
es_trigger = 0
best_val_f1 = -1.0
ckpt_path = train_file + '_ner_' + mymodel + "." + str(epochs) + '.pt'

model = model.to(device)

print("Train loop")
epochloop = tqdm(range(epochs), desc='Training')
for e in epochloop:
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for feature, target in trainloader:
        feature, target = feature.to(device), target.to(device)

        optim.zero_grad()
        out = model(feature)                                   # (N, L, C)
        loss = criterion(out.reshape(-1, num_classes), target.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        train_loss += loss.item()
        predicted = predict(out)
        mask = target != 0
        correct += ((predicted == target) & mask).sum().item()
        total += mask.sum().item()

    train_loss /= len(trainloader)
    train_acc = correct / max(total, 1)

    val_loss, val_acc, v_true, v_pred = evaluate(model, valloader)
    val_f1 = f1_score(v_true, v_pred, mode=MODE, scheme=SCHEME)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)

    epochloop.write(f'Epoch {e + 1}/{epochs} | Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} | '
                    f'Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f} Val F1 (seqeval): {val_f1:.3f}')

    # Sélection du modèle et early stopping sur le F1 seqeval de validation
    if val_f1 > best_val_f1:
        torch.save(model.state_dict(), ckpt_path)
        best_val_f1 = val_f1
        es_trigger = 0
    else:
        epochloop.write(f'[WARNING] Val F1 did not improve (best: {best_val_f1:.3f})')
        es_trigger += 1

    if es_trigger >= es_limit:
        epochloop.write(f'Early stopped at Epoch-{e + 1}')
        history['epochs'] = e + 1
        break

# --------------------------------------------------------------------------- #
# Test (avec le meilleur checkpoint)
# --------------------------------------------------------------------------- #
model.load_state_dict(torch.load(ckpt_path, map_location=device))
test_loss, test_acc, t_true, t_pred = evaluate(model, testloader)
print(f'Best val F1: {best_val_f1:.4f} | Test token accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}')

print(f'\n=== seqeval ({MODE}, {SCHEME.__name__}) ===')
print(classification_report(t_true, t_pred, mode=MODE, scheme=SCHEME, digits=4))
print('=== seqeval (conlleval, mode par défaut) ===')
print(classification_report(t_true, t_pred, digits=4))

# --------------------------------------------------------------------------- #
# Courbes d'apprentissage
# --------------------------------------------------------------------------- #
embedding_model_names = {
    EmbModel.FastText: 'FastText',
    EmbModel.W2V_CBow: 'Word2Vec CBow',
    EmbModel.W2V_Skipgram: 'Word2Vec Skipgram',
}
epochs_range = range(1, len(history['train_loss']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
ax1.plot(epochs_range, history['val_loss'], label='Validation Loss', marker='o')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title(f"Loss — {embedding_model_names[emb_model]}, lr={lr:g}, model={mymodel.lower()}")
ax1.legend()
ax1.grid(True)

ax2.plot(epochs_range, history['val_f1'], label='Validation F1 (seqeval)', marker='o', color='tab:green')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('F1')
ax2.set_title('Validation F1 (entités)')
ax2.legend()
ax2.grid(True)

Path('results').mkdir(exist_ok=True)
image_path = Path('results/learning_curve_loss.png')
suffix = 1
while image_path.exists():
    image_path = Path(f'results/learning_curve_loss_{suffix}.png')
    suffix += 1

plt.tight_layout()
plt.savefig(image_path)
plt.show()

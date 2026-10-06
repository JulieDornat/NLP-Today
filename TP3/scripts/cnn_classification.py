import pandas as pd
import numpy as np
import torch
import sys
from torch import nn
from torch.optim import Adam
from collections import Counter
from tqdm import tqdm
from torch.utils.data  import TensorDataset, DataLoader
from sklearn.metrics import classification_report
import torch.nn.functional as F
import argparse
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from seqeval.metrics import classification_report as cls_report, f1_score
from seqeval.scheme import IOB2

from torch.autograd import Variable

from enum import Enum
class EmbModel(Enum):
    FastText = 0
    W2V_CBow = 1
    W2V_Skipgram = 2

sequence_length = 128
max_vocab = 5000

parser = argparse.ArgumentParser()
parser.add_argument("--model",default='cnn',type=str,help="The kind of model (lstm or cnn -- default: lstm).",required=True)
parser.add_argument("--emb-model",default='0',type=int,help="0:FastText, 1: Word2Vec CBow, 2: Word2Vec Skipgram -- default: 0",required=True)
parser.add_argument("--emb-model-path",default='output/fattest_embeddings.bin',type=str,help="Path of the embedder -- default: output/fattest_embeddings.bin",required=True)
parser.add_argument("--train",default='',type=str,help="Training data in cnll format",required=True)
parser.add_argument("--valid",default='',type=str,help="Validation (valid or dev) data in cnll format",required=True)
parser.add_argument("--test",default='',type=str,help="Evaluation data in cnll format",required=True)
parser.add_argument("--epochs",default=1,type=int,help="Number of epoch")
parser.add_argument("--lr",default=0.0001,type=float,help="Learning rate")

args = parser.parse_args()

train_file = args.train # train file in csv format
valid_file = args.valid # dev/valid file in csv format
test_file = args.test # test file in csv format
#mymodel = "lstm" # cnn
mymodel = args.model # cnn or lstm
emb_model = EmbModel(args.emb_model)
emb_model_path = args.emb_model_path
lr = args.lr

epochs = args.epochs

print(emb_model)

import nltk
from nltk.corpus import stopwords

# Run this once if you haven't downloaded them yet
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('french'))

def read_conll(path):
    print("enterred read")
    sentences, labels = [], []
    tokens, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":              # blank line = sentence boundary
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
                continue
            parts = line.split()                # whitespace split (works here)
            if len(parts) < 5:                  # skip malformed/comment lines
                continue
            # Remove STOPWORDS
            # if parts[1].lower() in STOPWORDS:
            #     continue
            tokens.append(parts[1])             # token
            tags.append(parts[4])               # tag
        if tokens:                              # flush last sentence if file
            sentences.append(tokens)            # doesn't end with blank line
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

class NERModelLSTM(nn.Module):
    def __init__(self, embedding_weights, output_size, hidden_size=128, n_layers=2, dropout=0.2 ):
        super(NERModelLSTM, self).__init__()
        self.name = "lstm"
        # embedding layer is useful to map input into vector representation
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=True, padding_idx=0)
        embedding_size = embedding_weights.shape[1]

        # LSTM layer preserved by PyTorch library
        # Add bidirectionnal: look at the next word help to classify the token.
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True, bidirectional=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # Linear layer for output
        self.fc = nn.Linear(hidden_size*2, output_size)

        # Sigmoid layer cz we will have binary classification
        # Removed (only for binary classification): self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # convert feature to long
        x = x.long()

        # map input to vector
        x = self.embedding(x)

        # pass forward to lstm
        o, _ =  self.lstm(x)

        # get last sequence output
        #o = o[:, -1, :]

        # apply dropout and fully connected layer
        o = self.dropout(o)
        o = self.fc(o)

        return o

class NERModelCNN(nn.Module):

    def __init__(self, embedding_weights, class_size, dropout=0.2):
        super(NERModelCNN, self).__init__()
        self.name = "cnn"

        C = class_size
        Ci = 1
        Co = 100
        Ks = [3,5,7]

         # embedding layer is useful to map input into vector representation
        self.embed = nn.Embedding.from_pretrained(embedding_weights, freeze=True, padding_idx=0)
        embedding_size = embedding_weights.shape[1]
        print("Embedding initialized")

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embedding_size), padding=(K // 2, 0)) for K in Ks])
        
        print("Convs initialized")
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(len(Ks) * Co, C)

        #if self.args.static:
            #self.embed.weight.requires_grad = False

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [i.permute(0,2,1) for i in x]

        x = torch.cat(x, dim=2)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


# Our data is a set of tokens grouped by sentences.
# Each sentence is separated by a new line.
# The previous code was loading textual data and performing textual classification (one label per sentence).
# We want to perform a NER task, i.e. token classification.
# We adapt the data loading and preprocessing to have a set of tokens with their corresponding labels (one label per token).
# We change the embedding model to use our pretrained embeddings (FastText and Word2Vec).


def pad_features(sequences, pad_id, seq_length=128):
    # features = np.zeros((len(reviews), seq_length), dtype=int)
    features = np.full((len(sequences), seq_length), pad_id, dtype=int)

    for i, row in enumerate(sequences):
       # if seq_length < len(row) then review will be trimmed
        features[i, :len(row)] = np.array(row)[:seq_length]

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
                # If the token is unknown
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

    #Merge B- and I- variants into a single entity tag (for example, B-PROC/I-PROC -> PROC).
    # def normalize_tag(tag):
    #     if tag.startswith(('B-', 'I-')):
    #         return tag[2:]
    #     return tag

    # train_labels = [[normalize_tag(tag) for tag in tags] for tags in train_labels]
    # valid_labels = [[normalize_tag(tag) for tag in tags] for tags in valid_labels]
    # test_labels = [[normalize_tag(tag) for tag in tags] for tags in test_labels]

    for name, sents in [('train', train_sentences), ('valid', valid_sentences), ('test', test_sentences)]:
        n_long = sum(len(s) > seq_length for s in sents)
        if n_long:
            print(f"[WARNING] {name}: {n_long}/{len(sents)} phrases > {seq_length} tokens (tronquées)")

    print("Building vocabulary and tag dictionaries...")
    
    # 1. World vocabulary (based on all splits)
    all_words = [word for sent in train_sentences + valid_sentences + test_sentences for word in sent]
    counter = Counter(all_words)
    vocab = sorted(counter, key=counter.get, reverse=True)
    if max_vocab != -1:
        vocab = vocab[:max_vocab]
        
    int2word = dict(enumerate(vocab, 2))
    int2word[0] = '<PAD>'
    int2word[1] = '<UNK>'
    word2int = {word: id for id, word in int2word.items()}

    # 2. Label vocabulary (Tags NER, ex: B-PER, I-PER, O...)
    all_tags = set(tag for tags_list in train_labels + valid_labels + test_labels for tag in tags_list)
    
    # Index 0 is reserved for label padding (or -100 for CrossEntropyLoss)
    tag2int = {tag: i + 1 for i, tag in enumerate(sorted(all_tags))}
    tag2int['<PAD>'] = 0  # Padding for labels

    print("Classes and indexes:")
    for label, index in sorted(tag2int.items(), key=lambda item: item[1]):
        if label == '<PAD>':
            continue
        print(f"{label}: {index}")

    print("Encoding sequences...")
    # Encodage des mots (Features)
    l_train_x = encode_sequences(train_sentences, word2int, pad_id=word2int['<PAD>'], seq_length=seq_length)
    l_valid_x = encode_sequences(valid_sentences, word2int, pad_id=word2int['<PAD>'], seq_length=seq_length)
    l_test_x  = encode_sequences(test_sentences, word2int, pad_id=word2int['<PAD>'], seq_length=seq_length)

    # Encodage des labels (Targets) - Doit avoir la même forme que x (batch_size, seq_length)
    l_train_y = encode_sequences(train_labels, tag2int, pad_id=tag2int['<PAD>'], seq_length=seq_length, is_target=True)
    l_valid_y = encode_sequences(valid_labels, tag2int, pad_id=tag2int['<PAD>'], seq_length=seq_length, is_target=True)
    l_test_y  = encode_sequences(test_labels, tag2int, pad_id=tag2int['<PAD>'], seq_length=seq_length, is_target=True)

    return l_train_x, l_train_y, l_valid_x, l_valid_y, l_test_x, l_test_y, word2int, tag2int


train_x,train_y,valid_x,valid_y,test_x,test_y,word2int, tag2int = load_and_preprocess_data(train_file,valid_file,test_file,sequence_length,max_vocab)
int2tag = {idx: tag for tag, idx in tag2int.items()}

# print out the shape
print('Feature Shapes:')
print('===============')
print('Train set: {}'.format(train_x.shape))
print('Validation set: {}'.format(valid_x.shape))
print('Test set: {}'.format(test_x.shape))


# define batch size
batch_size = 128

# create tensor datasets
print("Create Tensor dataset")
trainset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
validset = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
testset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# create dataloaders
trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
valloader = DataLoader(validset, shuffle=True, batch_size=batch_size)
testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hidden_size = 128
n_layers = 1
dropout=0.25
max_document_length = sequence_length  # each sentence has until 100 words
#seed = 1
num_classes = len(tag2int)
#pool_size = 2
#n_filters = 128
#filter_sizes = [3, 8]

print("Load embeddings")
kv = load_embeddings(emb_model, emb_model_path)
embedding_weights = build_embedding_matrix(kv, word2int) 
del kv

# model initialization
print("Initialize model")
model = None
if mymodel == 'lstm':
    model = NERModelLSTM(embedding_weights, output_size=num_classes, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
if mymodel == 'cnn':
    model = NERModelCNN(embedding_weights, num_classes)
#model = SentimentModelCNN(vocab_size, embedding_size, n_filters, filter_sizes, pool_size, hidden_size, num_classes, sequence_length, dropout_keep_prob)
print(model)


# training config
#criterion = nn.BCELoss()  # we use BCELoss cz we have binary classification problem

# --------------------------------------------------------------------------- #
# Loss with smooth class weights (outside padding)
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


print_every = 1
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'epochs': epochs
}
es_limit = 5


# train loop
model = model.to(device)

epochloop = tqdm(range(epochs), position=0, desc='Training', leave=True)

# early stop trigger
es_trigger = 0
val_loss_min = torch.inf

print("Train loop")
for e in epochloop:

    #################
    # training mode #
    #################

    model.train()

    train_loss = 0
    total_correct = 0
    total_tokens = 0
    train_acc = 0

    for id, (feature, target) in enumerate(trainloader):
        # add epoch meta info
        print(f'Training batch {id}/{len(trainloader)}')
        epochloop.set_postfix_str(f'Training batch {id}/{len(trainloader)}')

        # move to device
        feature, target = feature.to(device), target.to(device)

        # reset optimizer
        optim.zero_grad()

        # forward pass -> shape : (batch_size, seq_length, num_classes)
        out = model(feature)

        predicted = torch.argmax(out, dim=-1)
        out_probs = []
        loss = criterion(out.view(-1, num_classes), target.view(-1))

        mask = target != 0
        correct = ((predicted == target) & mask).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0.0
        
        train_acc += acc
        total_correct += correct
        total_tokens += total
        train_loss += loss.item()

        loss.backward()

        # clip grad
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # update optimizer
        optim.step()

        # free some memory
        del feature, target, predicted, out

    history['train_loss'].append(train_loss / len(trainloader))
    history['train_acc'].append(total_correct / total_tokens if total_tokens > 0 else 0.0)

    ####################
    # validation model #
    ####################

    model.eval()

    val_loss = 0
    val_acc = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for id, (feature, target) in enumerate(valloader):
            # add epoch meta info
            epochloop.set_postfix_str(f'Validation batch {id}/{len(valloader)}')

            # move to device
            feature, target = feature.to(device), target.to(device)

            # forward pass
            out = model(feature)
            loss = criterion(out.view(-1, num_classes), target.view(-1))

            predicted = torch.argmax(out, dim=-1)

            mask = target != 0
            correct = ((predicted == target) & mask).sum().item()
            total = mask.sum().item()
            acc = correct / total if total > 0 else 0.0

            val_acc += acc
            val_loss += loss.item()
            total_correct += correct
            total_tokens += total

            # free some memory
            del feature, target, predicted, out

        history['val_loss'].append(val_loss / len(valloader))
        history['val_acc'].append(total_correct / total_tokens if total_tokens > 0 else 0.0)

    # reset model mode
    model.train()

    # add epoch meta info
    epochloop.set_postfix_str(f'Val Loss: {val_loss / len(valloader):.3f} | Val Acc: {val_acc / len(valloader):.3f}')

    # print epoch
    if (e+1) % print_every == 0:
        epochloop.write(f'Epoch {e+1}/{epochs} | Train Loss: {train_loss / len(trainloader):.3f} Train Acc: {train_acc / len(trainloader):.3f} | Val Loss: {val_loss / len(valloader):.3f} Val Acc: {val_acc / len(valloader):.3f}')
        epochloop.update()

    # save model if validation loss decrease
    if val_loss / len(valloader) <= val_loss_min:
        torch.save(model.state_dict(), train_file + '_sentiment_'+ mymodel + "." + str(epochs) + '.pt')
        val_loss_min = val_loss / len(valloader)
        es_trigger = 0
    else:
        epochloop.write(f'[WARNING] Validation loss did not improved ({val_loss_min:.3f} --> {val_loss / len(valloader):.3f})')
        es_trigger += 1

    # force early stop
    if es_trigger >= es_limit:
        epochloop.write(f'Early stopped at Epoch-{e+1}')
        # update epochs history
        history['epochs'] = e+1
        break

# --------------------------------------------------------------------------- #
# Evaluation utility functions (seqeval)
# --------------------------------------------------------------------------- #
def predict(out):
    """argmax by excluding the <PAD> class (index 0)."""
    out = out.detach().clone()
    out[..., 0] = float('-inf')
    return out.argmax(dim=-1)


def decode(predicted, target):
    """Tensors (N, L) -> two lists of lists of tags (one list per sentence), without padding."""
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


# test loop
model.eval()

# metrics
test_loss = 0
test_acc = 0
correct = 0
total = 0

all_target = []
all_predicted = []

testloop = tqdm(testloader, leave=True, desc='Inference')
with torch.no_grad():
    for feature, target in testloop:
        feature, target = feature.to(device), target.to(device)

        out = model(feature)
        loss = criterion(out.view(-1, num_classes), target.view(-1))
        test_loss += loss.item()

        predicted = predict(out)

        mask = target != 0
        correct += ((predicted == target) & mask).sum().item()
        total = mask.sum().item()
        # acc = correct / total if total > 0 else 0.0
        # test_acc += acc

        t, p = decode(predicted, target)
        all_target.extend(t)
        all_predicted.extend(p)

        # valid_indices = mask.view(-1)
        # flat_target = target.view(-1)
        # flat_predicted = predicted.view(-1)

        # all_target.extend(flat_target[valid_indices].cpu().numpy())
        # all_predicted.extend(flat_predicted[valid_indices].cpu().numpy())

    print(f'Accuracy: {correct/total if total > 0 else 0.0:.4f}, Loss: {test_loss/len(testloader):.4f}')

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

# Exclude the padding label (0) and report all actual NER classes.
print("=== Sklearn evaluation ===")
flat_target = [tag for sentence in all_target for tag in sentence]
flat_predicted = [tag for sentence in all_predicted for tag in sentence]
labels_to_eval = [tag for tag in int2tag.values() if tag != '<PAD>']

print(classification_report(flat_target, flat_predicted, labels=labels_to_eval, zero_division=0))

print("=== Seqeval evaluation ===")
print(cls_report(all_target, all_predicted, scheme=IOB2, mode='strict'))
print(f"Micro-F1 (strict): {f1_score(all_target, all_predicted, scheme=IOB2, mode='strict'):.4f}")

# --- Display learning curve ---
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
epochs_range = range(1, len(history['train_loss']) + 1)

plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
plt.plot(epochs_range, history['val_loss'], label='Validation Loss', marker='o')

plt.xlabel('Epochs')
plt.ylabel('Loss')
embedding_model_names = {
    EmbModel.FastText: 'FastText',
    EmbModel.W2V_CBow: 'Word2Vec CBow',
    EmbModel.W2V_Skipgram: 'Word2Vec Skipgram',
}
plt.title(
    f"Learning curve (Loss by Epoch) — {embedding_model_names[emb_model]}, "
    f"lr={lr:g}, model={mymodel.lower()}"
)
plt.legend()
plt.grid(True)

# Save learning curve image

from pathlib import Path

image_path = Path('results/learning_curve_loss.png')
suffix = 1
while image_path.exists():
    image_path = Path(f'results/learning_curve_loss_{suffix}.png')
    suffix += 1

plt.savefig(image_path)
plt.show()






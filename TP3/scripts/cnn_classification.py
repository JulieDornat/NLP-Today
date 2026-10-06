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
valid_file = args.test # dev/valid file in csv format
test_file = args.valid # test file in csv format
#mymodel = "lstm" # cnn
mymodel = args.model # cnn or lstm
emb_model = EmbModel(args.emb_model)
emb_model_path = args.emb_model_path
lr = args.lr

epochs = args.epochs

print(emb_model)
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
            tokens.append(parts[1])             # token
            tags.append(parts[4])               # tag
        if tokens:                              # flush last sentence if file
            sentences.append(tokens)            # doesn't end with blank line
            labels.append(tags)
    return sentences, labels

class NERModelLSTM(nn.Module):
    def __init__(self, output_size, hidden_size=128, n_layers=2, dropout=0.2, emb_model: EmbModel = EmbModel.FastText, emb_model_path = "output/fattest_embeddings.bin" ):
        super(NERModelLSTM, self).__init__()
        self.name = "lstm"
        # embedding layer is useful to map input into vector representation
        if emb_model == EmbModel.FastText:
            loaded_emb_model = load_facebook_vectors(emb_model_path)
        elif emb_model == EmbModel.W2V_CBow or emb_model == EmbModel.W2V_Skipgram:
            # emb_model_path = extracted_data/word2vec_output/embeddings/w2v_cbow_med.vec
            loaded_emb_model = KeyedVectors.load_word2vec_format(emb_model_path, binary=False)
        else:
            raise Exception("Unknown embedder.")

        vocab_size = len(loaded_emb_model.key_to_index)
        embedding_size = loaded_emb_model.vector_size
        weights = torch.FloatTensor(loaded_emb_model.vectors)
        self.embedding = nn.Embedding(vocab_size, embedding_size).from_pretrained(weights)

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

    def __init__(self,class_size, dropout=0.2, emb_model: EmbModel = EmbModel.FastText, emb_model_path = "output/fattest_embeddings.bin"):
        super(NERModelCNN, self).__init__()
        self.name = "cnn"

        C = class_size
        Ci = 1
        Co = 100
        Ks = [3,5,7]

         # embedding layer is useful to map input into vector representation
        if emb_model == EmbModel.FastText:
            loaded_emb_model = load_facebook_vectors(emb_model_path)
        elif emb_model == EmbModel.W2V_CBow or emb_model == EmbModel.W2V_Skipgram:
            # emb_model_path = extracted_data/word2vec_output/embeddings/w2v_cbow_med.vec
            loaded_emb_model = KeyedVectors.load_word2vec_format(emb_model_path, binary=False)
        else:
            raise Exception("Unknown embedder.")
        vocab_size = len(loaded_emb_model.key_to_index)
        embedding_size = loaded_emb_model.vector_size
        weights = torch.FloatTensor(loaded_emb_model.vectors)
        self.embed = nn.Embedding(vocab_size, embedding_size).from_pretrained(weights)
        print("Embedding initialized")

        V = vocab_size
        D = embedding_size
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


#dropout_keep_prob = 0.5
max_document_length = sequence_length  # each sentence has until 100 words
#seed = 1
num_classes = len(tag2int)
#pool_size = 2
#n_filters = 128
#filter_sizes = [3, 8]

# model initialization
print("Initialize model")
model = None
if mymodel == 'lstm':
    model = NERModelLSTM(output_size=num_classes, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout, emb_model_path=emb_model_path, emb_model=emb_model)
if mymodel == 'cnn':
    model = NERModelCNN(num_classes, emb_model_path=emb_model_path, emb_model=emb_model)
#model = SentimentModelCNN(vocab_size, embedding_size, n_filters, filter_sizes, pool_size, hidden_size, num_classes, sequence_length, dropout_keep_prob)
print(model)


# training config
#criterion = nn.BCELoss()  # we use BCELoss cz we have binary classification problem

# IMPROVEMENT: Smooth weights to prevent majority class dominance.
from sklearn.utils.class_weight import compute_class_weight
y_train_flat = np.concatenate([seq for seq in train_y])
y_train_flat_no_pad = y_train_flat[y_train_flat != 0] # we remove the padding class

classes_unique = np.unique(y_train_flat)
raw_weights = compute_class_weight('balanced', classes=classes_unique, y=y_train_flat)

smoothed_weights = np.sqrt(raw_weights) 
smoothed_weights = smoothed_weights / np.mean(smoothed_weights)

weights_tensor = torch.FloatTensor(smoothed_weights).to(device)
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
        train_loss += loss.item()

        loss.backward()

        # clip grad
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # update optimizer
        optim.step()

        # free some memory
        del feature, target, predicted, out

    history['train_loss'].append(train_loss / len(trainloader))
    history['train_acc'].append(train_acc / len(trainloader))

    ####################
    # validation model #
    ####################

    model.eval()

    val_loss = 0
    val_acc = 0

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

            # free some memory
            del feature, target, predicted, out

        history['val_loss'].append(val_loss / len(valloader))
        history['val_acc'].append(val_acc / len(valloader))

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

# test loop
model.eval()

# metrics
test_loss = 0
test_acc = 0

all_target = []
all_predicted = []

testloop = tqdm(testloader, leave=True, desc='Inference')
with torch.no_grad():
    for feature, target in testloop:
        feature, target = feature.to(device), target.to(device)

        out = model(feature)
        loss = criterion(out.view(-1, num_classes), target.view(-1))
        test_loss += loss.item()

        predicted = torch.argmax(out, dim=-1)

        mask = target != 0
        correct = ((predicted == target) & mask).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0.0
        test_acc += acc

        valid_indices = mask.view(-1)
        flat_target = target.view(-1)
        flat_predicted = predicted.view(-1)

        all_target.extend(flat_target[valid_indices].cpu().numpy())
        all_predicted.extend(flat_predicted[valid_indices].cpu().numpy())

    print(f'Accuracy: {test_acc/len(testloader):.4f}, Loss: {test_loss/len(testloader):.4f}')


print(classification_report(all_target, all_predicted))

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

image_path = Path('learning_curve_loss.png')
suffix = 1
while image_path.exists():
    image_path = Path(f'learning_curve_loss_{suffix}.png')
    suffix += 1

plt.savefig(image_path)
plt.show()






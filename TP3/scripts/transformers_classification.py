#!/usr/bin/env python
# coding: utf-8

import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import TrainingArguments, AutoTokenizer, EvalPrediction, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
import argparse
from datasets import Dataset, Features, Sequence, Value, ClassLabel

import time
from seqeval.metrics import (
    precision_score as seq_precision,
    recall_score as seq_recall,
    f1_score as seq_f1,
    classification_report as seq_report,
)


print()
print("starting")
print()

parser = argparse.ArgumentParser(description='Finetuning BERT kind models for multi-class classification')
parser.add_argument('--model', type=str, help='Huggingface BERT model to be called', required=True)
parser.add_argument('--train', type=str, help='Train processed input saved directory', required=True)
parser.add_argument('--test', type=str, help='Test processed input saved directory', required=True)
parser.add_argument('--valid', type=str, help='Validation processed input saved directory', required=True)
parser.add_argument('--epochs',  type=int, help='Numbers of epochs (default 5)', default=5)
parser.add_argument('--batch',  type=int, help='batch size (default 8)', default=8)
parser.add_argument('--max_len',  type=int, help='maximum text length (default 128)', default=128)
parser.add_argument('--lr',  type=float, help='learning rate (default 1e-05)', default=1e-05)



args = parser.parse_args()

print("parsed args")

## New code

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
# "C:/Users/marin/OneDrive - Université Paris-Saclay/P5/NLP Today/NLP-Today/TP3/QUAERO_FrenchMed/MEDLINE/MEDLINEdev_layer1_ID.conll
train_tokens, train_tags = read_conll(args.train)
test_tokens, test_tags = read_conll(args.test)
dev_tokens, dev_tags = read_conll(args.valid)

print(train_tokens)
print(dev_tags)

for name, toks, tags in [("train", train_tokens, train_tags),
                          ("dev",   dev_tokens,   dev_tags),
                          ("test",  test_tokens,  test_tags)]:
    assert len(toks) == len(tags), f"{name}: sentence count mismatch"
    bad = [i for i, (t, g) in enumerate(zip(toks, tags)) if len(t) != len(g)]
    print(f"{name}: {len(toks)} sentences, {sum(map(len, toks))} tokens, "
          f"mismatched sentences: {len(bad)}")
    if bad:
        i = bad[0]
        print("  first mismatch:", i, "tokens:", len(toks[i]), "tags:", len(tags[i]))


###


bert_model = args.model
MAX_LEN = args.max_len
TRAIN_BATCH_SIZE = args.batch
VALID_BATCH_SIZE = 4
EPOCHS = args.epochs
LEARNING_RATE = args.lr



# building tag vocabulary
all_tags = sorted({t for split in (train_tags, dev_tags, test_tags) for s in split for t in s})
tag2id = {t: i for i, t in enumerate(all_tags)}
id2tag = {i: t for t, i in tag2id.items()}
print(f"{len(all_tags)} tags: {all_tags}")
print(tag2id)


# building HF datasets and tokenize with alignment


features = Features({
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(ClassLabel(names=all_tags))
})
print("features")
print(features)

def make_ds(tokens, tags):
    return Dataset.from_dict(
        {
            "tokens": tokens, 
            "ner_tags": [[tag2id[t] for t in s] for s in tags]
        }
    )

train_ds = make_ds(train_tokens, train_tags)
dev_ds = make_ds(dev_tokens, dev_tags)
test_ds = make_ds(test_tokens, test_tags)

print("datasets")
print(train_ds)
print(dev_ds)
print(test_ds)

tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=True)

def tokenize_and_align(examples):
    tok = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, max_length=MAX_LEN, padding=False)
    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tok.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # set -100 as label for these special tokens
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                # add the corresponding label
                label_ids.append(tags[word_idx])
            else:
                label_ids.append(-100) # -100 if label all tokens is false
            prev_word_idx = word_idx
        all_labels.append(label_ids)
    tok['labels'] = all_labels
    return tok

# applies tokenization to all splits
train_ds = train_ds.map(tokenize_and_align, batched=True)
dev_ds   = dev_ds.map(tokenize_and_align,   batched=True)
test_ds  = test_ds.map(tokenize_and_align,  batched=True)

# input_ids are the subtoken IDs to feed into BERT's embedding layer
# attention_mask tells the model which positions to attend to. 1 for real tokens, 0 for padding
# labels has the IDs per subtoken, -100 at those which we can ignore
cols = ["input_ids", "attention_mask", "labels"] # we need these tensors for the BERT model

train_ds.set_format("torch", columns=cols)
dev_ds.set_format("torch",   columns=cols)
test_ds.set_format("torch",  columns=cols)

print("After tokenization:")
print("  train:", len(train_ds))
print("  dev:  ", len(dev_ds))
print("  test: ", len(test_ds))
print("  sample:", {k: v[:5].tolist() for k, v in train_ds[0].items()})



## TRAINING

# architecture composed of:
# 1. embedding layer - lookup table mapping each token ID to a vector
# 2. 12 transformer encoder layers: attention weights, feedforward weights, layer norms
# 3. Classification head: a small linear layer from hidden_size=768 to num_labels=21. around 16k params


## define metrics


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=-1)
    y_true, y_pred = [], []
    for pred_row, label_row in zip(preds, p.label_ids):
        keep = label_row != -100
        y_true.append([id2tag[l] for l, k in zip(label_row, keep) if k])
        y_pred.append([id2tag[pr] for pr, k in zip(pred_row, keep) if k])
    return {
        "precision": seq_precision(y_true, y_pred),
        "recall":    seq_recall(y_true, y_pred),
        "f1":        seq_f1(y_true, y_pred),
    }

# load the model

# this loads the pretrained encoder and attaches a fresh 21 tag classification head

model = AutoModelForTokenClassification.from_pretrained(
    bert_model,
    num_labels=len(all_tags),
    id2label=id2tag,
    label2id=tag2id,
)
print(f"Loaded {bert_model}, num_labels = {len(all_tags)}")

# define the training arguments

training_args = TrainingArguments(
    output_dir=f"ner_{bert_model.replace('/', '_')}",
    evaluation_strategy="epoch",     # no choice if transformers >= 4.46
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    load_best_model_at_end=True, # at end keep check poing with highest dev F1
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=1,
    report_to="none",
)

# trainer and Data Collator

# What they do: 
# DataCollatorForTokenClassification — pads input_ids with 0, attention_mask with 0, labels with -100, per batch.
# Trainer — runs the standard HF training loop: forward → loss → backprop → step, with periodic eval.

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,             # processing_class= if get a deprecation warning
)

# starting training

start = time.time()
trainer.train()
print(f"\nTraining time: {time.time() - start:.1f} s\n")

print("=== Validation ===")
print(trainer.evaluate(dev_ds))
print("=== Test ===")
print(trainer.evaluate(test_ds))


# full per class report with seqeval
out   = trainer.predict(test_ds)
preds = np.argmax(out.predictions, axis=-1)

y_true, y_pred = [], []
for pred_row, label_row in zip(preds, out.label_ids):
    keep = label_row != -100
    y_true.append([id2tag[l] for l, k in zip(label_row, keep) if k])
    y_pred.append([id2tag[pr] for pr, k in zip(pred_row, keep) if k])

print("\n=== Full seqeval report (test) ===")
print(seq_report(y_true, y_pred))

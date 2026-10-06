import subprocess
import itertools
import pandas as pd
import os
import re

TRAIN_DATA_PATH = "QUAERO_FrenchMed/MEDLINE/MEDLINEtrain_layer1_ID.conll"
VALID_DATA_PATH = "QUAERO_FrenchMed/MEDLINE/MEDLINEdev_layer1_ID.conll"
TEST_DATA_PATH = "QUAERO_FrenchMed/MEDLINE/MEDLINEtest_layer1_ID.conll"

DATA_DIR = "extracted_data"
OUT_DIR = os.path.join(DATA_DIR, "word2vec_output")
EMB_DIR = os.path.join(OUT_DIR, "embeddings")

emb_model_path = [f"./output/fasttext_med.bin",
                  f"{EMB_DIR}/w2v_cbow_med.vec",
                  f"{EMB_DIR}/w2v_skipgram_med.vec"]


models = ['cnn', 'lstm']
learning_rates = [1e-5, 5e-5, 1e-4]
index = [0, 1, 2]
results = []

def extract_f1(output):
    """Extract the last explicitly reported F1 score from script output."""
    matches = re.findall(
        r"\bf1(?:[-_\s]?score)?\s*[:=]\s*(\d+(?:\.\d+)?)%?",
        output,
        flags=re.IGNORECASE,
    )
    if matches:
        return float(matches[-1])

    # Support the weighted-average row of a sklearn classification report.
    rows = re.findall(
        r"(?im)^\s*weighted\s+avg\s+\d+(?:\.\d+)?\s+"
        r"\d+(?:\.\d+)?\s+(\d+(?:\.\d+)?)",
        output,
    )
    return float(rows[-1]) if rows else None


# Pair each embedding path with its corresponding embedding index.
embeddings = list(zip(emb_model_path, index))
grid = list(itertools.product(models, learning_rates, embeddings))
results = []

print(f"Launch {len(grid)} configurations...")

for model, lr, (emb_path, idx) in grid:
    print(f"\n--- Test : Modèle={model}, LR={lr} Emb={emb_path} ---")
    
    cmd = [
        "python", "scripts/cnn_classification.py", 
        "--model", model,
        "--emb-model", str(idx),
        "--emb-model-path", emb_path,
        "--train", TRAIN_DATA_PATH,
        "--valid", VALID_DATA_PATH,
        "--test", TEST_DATA_PATH,
        "--epochs", "50",
        "--lr", str(lr),
    ]
    print(cmd)
    # Script execution and retrieve output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        f1 = extract_f1(result.stdout)
        if f1 is not None:
            results.append({"embedding": emb_path, "model": model, "lr": lr, "f1": f1})
        else:
            print("Could not find an F1 score in the script output.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution : {e.stderr}")

print("\nBest configuration for each embedding:")
for emb_path, _ in embeddings:
    embedding_results = [r for r in results if r["embedding"] == emb_path]
    if embedding_results:
        best = max(embedding_results, key=lambda r: r["f1"])
        print(
            f"{emb_path}: model={best['model']}, learning rate={best['lr']}, "
            f"F1={best['f1']}"
        )
    else:
        print(f"{emb_path}: no F1 results available")
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
learning_rates = [1e-5, 5e-5, 1e-4, 1e-3]
index = [0, 1, 2]
results = []

def extract_f1(output):
    """Extract the last explicitly reported F1 score from script output."""
    match = re.search(r"Micro-F1\s*\(strict\)\s*:\s*(\d+(?:\.\d+)?)",
                      output, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Support the weighted-average row of a seqeval classification report.
    seqeval_section = output.split("=== Seqeval evaluation ===")
    if len(seqeval_section) > 1:
        micro_match = re.search(r"micro avg\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)", seqeval_section[1])
        if micro_match:
            return float(micro_match.group(1))
            
    return None


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

print("\n========================================")
print("Best configuration for each embedding (based on Seqeval Micro-F1):")
print("========================================")
for emb_path, _ in embeddings:
    embedding_results = [r for r in results if r["embedding"] == emb_path]
    if embedding_results:
        best = max(embedding_results, key=lambda r: r["seqeval_f1"])
        print(
            f"Embedding: {emb_path}\n"
            f"  -> Model: {best['model']}\n"
            f"  -> Learning Rate: {best['lr']}\n"
            f"  -> Seqeval Micro-F1: {best['seqeval_f1']:.4f}\n"
        )
    else:
        print(f"{emb_path}: no results available\n")

# Optionnel : Afficher la toute meilleure configuration globale toutes catégories confondues
if results:
    absolute_best = max(results, key=lambda r: r["seqeval_f1"])
    print("========================================")
    print(f"🏆 ABSOLUTE BEST CONFIGURATION (Seqeval):")
    print(f"   Model: {absolute_best['model']}")
    print(f"   Embedding: {absolute_best['embedding']}")
    print(f"   Learning Rate: {absolute_best['lr']}")
    print(f"   Seqeval Micro-F1: {absolute_best['seqeval_f1']:.4f}")
    print("========================================")
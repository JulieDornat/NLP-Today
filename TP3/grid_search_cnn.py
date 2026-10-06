import subprocess
import itertools
import pandas as pd
import os

TRAIN_DATA_PATH = "QUAERO_FrenchMed/EMEA/EMEAtrain_layer1_ID.conll"
VALID_DATA_PATH = "QUAERO_FrenchMed/EMEA/EMEAdev_layer1_ID.conll"
TEST_DATA_PATH = "QUAERO_FrenchMed/EMEA/EMEAtest_layer1_ID.conll"

DATA_DIR = "extracted_data"
OUT_DIR = os.path.join(DATA_DIR, "word2vec_output")
EMB_DIR = os.path.join(OUT_DIR, "embeddings")

emb_model_path = [f"./output/fasttext_med.bin",
                  f"{EMB_DIR}/w2v_cbow_med.vec",
                  f"{EMB_DIR}/w2v_skipgram_med.vec "]


models = ['cnn', 'lstm']
learning_rates = [1e-3, 5e-4, 1e-4]
index = [0, 1, 2]
results = []

# Generate all possible combinaisons
grid = list(itertools.product(models, learning_rates, emb_model_path, index))

print(f"Launch {len(grid)} configurations...")

for model, lr, emb_path, idx in grid:
    print(f"\n--- Test : Modèle={model}, LR={lr} Emb={emb_path} ---")
    
    cmd = [
        "python", "scripts/cnn_classification.py", 
        "--model", model,
        "--emb-model", str(idx),
        "--emb-model-path", emb_path,
        "--train", TRAIN_DATA_PATH,
        "--valid", VALID_DATA_PATH,
        "--test", TEST_DATA_PATH,
        "--epochs", "20"
        "--lr", str(lr)
    ]
    
    # Exécution du script et récupération de la sortie
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        # Vous pouvez parser la sortie standard ici pour récupérer l'accuracy de test finale si vous le souhaitez
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution : {e.stderr}")
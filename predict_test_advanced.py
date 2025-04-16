
import torch
import h5py
import numpy as np
import pandas as pd
from train_model_advanced import InsolesMocapClassifier

# Chargement des données test
with h5py.File('test_insoles.h5', 'r') as f:
    test_insoles = f['my_data'][:]
with h5py.File('test_mocap.h5', 'r') as f:
    test_mocap = f['my_data'][:]

def preprocess(seq, shape):
    seq = seq.reshape(shape)
    seq = torch.tensor(seq, dtype=torch.float32)
    seq = torch.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    seq = torch.clamp(seq, -10, 10)
    seq = (seq - seq.mean()) / (seq.std() + 1e-8)
    return seq

# Charger le modèle entraîné avancé
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InsolesMocapClassifier()
model.load_state_dict(torch.load('model_best_advanced.pth', map_location=device))
model.to(device)
model.eval()

# Prédictions
preds = []
with torch.no_grad():
    for i in range(len(test_insoles)):
        try:
            x_insole = preprocess(test_insoles[i], (100, 50)).unsqueeze(0).to(device)
            x_mocap = preprocess(test_mocap[i], (100, 129)).unsqueeze(0).to(device)
            output = model(x_insole, x_mocap)
            pred = torch.argmax(output, dim=1).item()
        except Exception as e:
            print(f"❌ Erreur sur l'élément {i}: {e}")
            pred = -1
        preds.append(pred)

# Correction si prédiction manquante
while len(preds) < 274:
    print("⚠️ Ajout d'une prédiction de secours pour compléter à 274.")
    preds.append(preds[-1] if preds else 0)

# Génération du fichier de soumission
submission = pd.DataFrame({'ID': np.arange(1, 275), 'Label': preds})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv généré avec succès avec 274 lignes.")

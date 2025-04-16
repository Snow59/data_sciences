# Projet Data Science - Classification de Mouvements (Semelles + Motion Capture)

## 🎯 Objectif
L'objectif du projet est de classifier des mouvements humains (13 classes comme marcher, squat, yoga...) à partir de deux types de données :
- `Semelles` : données de capteurs dans des semelles connectées (forme : 100x50)
- `Motion Capture` : positions 3D de points corporels (forme : 100x129)

## 🛠️ Données
- **train_insoles.h5** : séquences de capteurs plantaires
- **train_mocap.h5** : données de capture de mouvement
- **train_labels.h5** : labels associés (0 à 12)
- **test_insoles.h5 / test_mocap.h5** : données de test (sans labels)

## 🧩 Architecture du modèle final (`advanced_v2`)
- 📥 2 entrées : insole (100x50) et mocap (100x129)
- 🧠 Traitement :
  - Conv1D (kernel=3) sur chaque séquence
  - LSTM bi-directionnel (2 couches, 384 neurones)
  - LayerNorm sur les sorties
  - Self-Attention pour pondérer temporellement
  - Fusion des deux branches par moyenne
  - Dense → ReLU → Dropout (0.5) → Output
- 🎓 Entraînement :
  - Optimiseur : Adam (lr=0.0005)
  - Perte : CrossEntropyLoss avec `class_weight`
  - Batch Size : **128**
  - Epochs : **30**
  - Scheduler et early stopping

## 🧪 Prétraitement
- Remplacement des NaN et Inf
- Clamp entre [-10, 10]
- Normalisation (z-score) séquence par séquence
- Data augmentation (petit bruit gaussien)

## 📊 Résultat
- **Validation Accuracy** : ~95.6%
- **Score Kaggle** : 🏅 **0.860**

## 📁 Fichiers
- `train_model_advanced_v2.py` → modèle complet
- `predict_test_advanced_v2.py` → prédictions sur données de test
- `submission.csv` → fichier au bon format (274 lignes)

## 🙋 Auteur
Hamza – IMT Nord Europe – 4ᵉ année Informatique & Télécommunications

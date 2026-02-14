# 📖 README.md

```markdown
# 🐗 RF-DETR Training Pipeline

Pipeline d'entraînement **RF-DETR** (Roboflow Detection Transformer) pour la détection d'objets multi-classes.

Compatible avec :
- 🖥️ **NVIDIA GPU** (CUDA) - RunPod, Lambda Labs, etc.
- 🍎 **Apple Silicon** (MPS) - Mac M1/M2/M3/M4
- 💻 **CPU** - Debug uniquement

---

## 📋 Table des matières

- [Installation](#-installation)
- [Configuration rapide](#-configuration-rapide)
- [Fichier CSV des datasets](#-fichier-csv-des-datasets)
- [Options de ligne de commande](#-options-de-ligne-de-commande)
- [Exemples d'utilisation](#-exemples-dutilisation)
- [Configuration par device](#-configuration-par-device)
- [Fichiers générés](#-fichiers-générés)
- [Architecture du projet](#-architecture-du-projet)
- [Troubleshooting](#-troubleshooting)

---

## 🔧 Installation

### Prérequis

- Python 3.9+
- PyTorch 2.0+
- Compte [Roboflow](https://roboflow.com) avec clé API

### Installation des dépendances

```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install rfdetr roboflow torch torchvision supervision
```

### Installation spécifique par plateforme

<details>
<summary>🖥️ NVIDIA GPU (CUDA)</summary>

```bash
# PyTorch avec CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Ou CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install rfdetr roboflow supervision
```
</details>

<details>
<summary>🍎 Mac Apple Silicon (MPS)</summary>

```bash
# PyTorch natif pour Apple Silicon
pip install torch torchvision

pip install rfdetr roboflow supervision
```
</details>

---

## ⚡ Configuration rapide

### 1. Créer le fichier CSV des datasets

```csv
name,workspace,project,version
sanglier,corbin-helms-wghhy,wild-boar-detection-0igwx,1
frelon,use-case-asian-hornet-detection,asian-hornet-video-detection,1
```

### 2. Obtenir votre clé API Roboflow

1. Connectez-vous à [Roboflow](https://app.roboflow.com)
2. Allez dans **Settings** → **API Keys**
3. Copiez votre clé API privée

### 3. Lancer l'entraînement

```bash
# Entraînement rapide (1 epoch pour test)
python train_rfdetr.py --api-key VOTRE_CLE --dataset sanglier --epochs 1

# Entraînement complet
python train_rfdetr.py --api-key VOTRE_CLE --dataset sanglier --epochs 100
```

---

## 📄 Fichier CSV des datasets

Le fichier CSV définit les datasets à télécharger depuis Roboflow.

### Format

| Colonne | Description | Exemple |
|---------|-------------|---------|
| `name` | Nom local du dataset | `sanglier` |
| `workspace` | Workspace Roboflow | `corbin-helms-wghhy` |
| `project` | Nom du projet | `wild-boar-detection-0igwx` |
| `version` | Version du dataset | `1` |

### Exemple complet

```csv
name,workspace,project,version
sanglier,corbin-helms-wghhy,wild-boar-detection-0igwx,1
frelon,use-case-asian-hornet-detection,asian-hornet-video-detection,1
cerf,mon-workspace,deer-detection,2
chat,mon-workspace,cat-detection,1
```

### Trouver les informations Roboflow

Sur la page de votre projet Roboflow, l'URL a ce format :
```
https://app.roboflow.com/WORKSPACE/PROJECT/VERSION
```

Exemple : `https://app.roboflow.com/corbin-helms-wghhy/wild-boar-detection-0igwx/1`
- **workspace** : `corbin-helms-wghhy`
- **project** : `wild-boar-detection-0igwx`
- **version** : `1`

---

## 🎛️ Options de ligne de commande

### Options principales

| Option | Description | Défaut |
|--------|-------------|--------|
| `--dataset` | Dataset à entraîner (ou `all`) | Auto si 1 seul |
| `--datasets-csv` | Chemin du fichier CSV | `datasets.csv` |
| `--api-key` | Clé API Roboflow | `$ROBOFLOW_API_KEY` |
| `--device` | Device de calcul | `auto` |
| `--epochs` | Nombre d'époques | Auto selon device |

### Options de modèle

| Option | Description | Valeurs | Défaut |
|--------|-------------|---------|--------|
| `--model-size` | Taille du modèle | `nano`, `small`, `base`, `large` | Auto |
| `--batch` | Taille du batch | 1-32 | Auto |
| `--resolution` | Résolution des images | 480-640 | Auto |
| `--lr` | Learning rate | 1e-5 à 1e-3 | `1e-4` |
| `--grad-accum` | Gradient accumulation | 1-16 | Auto |
| `--workers` | Workers pour data loading | 0-16 | Auto |

### Options spéciales

| Option | Description |
|--------|-------------|
| `--list-datasets` | Affiche les datasets du CSV |
| `--predict` | Mode prédiction |
| `--model` | Chemin du modèle (prédiction) |
| `--image` | Image pour prédiction |

---

## 📝 Exemples d'utilisation

### Entraînement

```bash
# Lister les datasets disponibles
python train_rfdetr.py --list-datasets

# Test rapide (1 epoch)
python train_rfdetr.py --api-key CLE --dataset sanglier --epochs 1

# Entraînement complet sur GPU
python train_rfdetr.py --api-key CLE --dataset sanglier --device cuda --epochs 100

# Entraînement avec paramètres custom
python train_rfdetr.py \
    --api-key CLE \
    --dataset sanglier \
    --device cuda \
    --model-size large \
    --batch 8 \
    --epochs 150 \
    --lr 5e-5

# Entraîner TOUS les datasets du CSV
python train_rfdetr.py --api-key CLE --dataset all

# Utiliser un fichier CSV différent
python train_rfdetr.py --api-key CLE --datasets-csv production.csv --dataset all
```

### Prédiction

```bash
# Prédiction avec modèle entraîné
python train_rfdetr.py --predict --model projet_detection_rfdetr/models/sanglier/checkpoint.pth

# Prédiction sur une image spécifique
python train_rfdetr.py --predict \
    --model projet_detection_rfdetr/models/sanglier/checkpoint.pth \
    --image mon_image.jpg

# Prédiction avec une URL
python train_rfdetr.py --predict \
    --model projet_detection_rfdetr/models/sanglier/checkpoint.pth \
    --image "https://example.com/image.jpg"
```

### Variables d'environnement

```bash
# Définir la clé API en variable d'environnement
export ROBOFLOW_API_KEY="votre_cle_api"

# Puis lancer sans --api-key
python train_rfdetr.py --dataset sanglier --epochs 100
```

---

## 🖥️ Configuration par device

Le script ajuste automatiquement les paramètres selon le matériel détecté.

### 🟢 NVIDIA GPU (CUDA)

| GPU | Model Size | Batch | Grad Accum | Resolution | Workers |
|-----|------------|-------|------------|------------|---------|
| H100 (80GB) | `large` | 16 | 1 | 560 | 8 |
| A100 (80GB) | `large` | 16 | 1 | 560 | 8 |
| A100 (40GB) | `large` | 8 | 2 | 560 | 8 |
| RTX 4090 (24GB) | `base` | 8 | 2 | 560 | 8 |
| RTX 3090 (24GB) | `base` | 4 | 4 | 560 | 8 |
| RTX 3080 (10GB) | `small` | 4 | 4 | 560 | 8 |
| GPU < 10GB | `nano` | 2 | 8 | 560 | 8 |

**Temps estimé** : ~2 min/epoch

### 🍎 Apple Silicon (MPS)

| Paramètre | Valeur |
|-----------|--------|
| Model Size | `nano` |
| Batch Size | 4 |
| Grad Accum | 4 |
| Resolution | 640 |
| Workers | 0 |
| Learning Rate | 1e-4 |

**Temps estimé** : ~10 min/epoch

> ⚠️ Le flag `PYTORCH_ENABLE_MPS_FALLBACK=1` est activé automatiquement pour les opérations non supportées par MPS.

### 💻 CPU

| Paramètre | Valeur |
|-----------|--------|
| Model Size | `nano` |
| Batch Size | 2 |
| Grad Accum | 8 |
| Resolution | 480 |
| Workers | 0 |
| Learning Rate | 1e-4 |

**Temps estimé** : ~60 min/epoch (non recommandé pour l'entraînement)

---

## 📁 Fichiers générés

### Structure du projet

```
projet_detection_rfdetr/
├── datasets/
│   ├── sanglier/
│   │   ├── train/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── _annotations.coco.json
│   │   ├── valid/
│   │   │   └── _annotations.coco.json
│   │   └── test/
│   │       └── _annotations.coco.json
│   └── frelon/
│       └── ...
│
├── models/
│   ├── sanglier/
│   │   ├── checkpoint.pth              # Dernier checkpoint
│   │   ├── checkpoint_best_total.pth   # Meilleur modèle
│   │   ├── class_mapping.json          # Mapping des classes
│   │   └── exports/
│   │       └── checkpoint.onnx         # Export ONNX
│   └── frelon/
│       └── ...
│
└── datasets.csv                        # Configuration des datasets
```

### Fichier `class_mapping.json`

Ce fichier est généré automatiquement avec chaque modèle entraîné :

```json
{
  "created_at": "2024-12-15T10:30:00.123456",
  "datasets_used": ["sanglier"],
  "classes": {
    "0": {
      "name": "sanglier",
      "dataset": "sanglier",
      "supercategory": "animal"
    }
  },
  "total_classes": 1
}
```

**Utilité** :
- Permet de retrouver le nom des classes lors de la prédiction
- Trace les datasets utilisés pour l'entraînement
- Facilite le déploiement sur d'autres machines

---

## 🏗️ Architecture du projet

### Modèles RF-DETR disponibles

| Modèle | Params | COCO mAP | Vitesse | Usage recommandé |
|--------|--------|----------|---------|------------------|
| `nano` | 3M | 48.0 | ⚡⚡⚡⚡ | Edge, Mobile, Tests |
| `small` | 12M | 52.0 | ⚡⚡⚡ | Edge, Jetson |
| `base` | 29M | 55.0 | ⚡⚡ | Production |
| `large` | 128M | 57.0 | ⚡ | Max précision |

### Pipeline d'entraînement

```
┌─────────────────┐
│  datasets.csv   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Download from   │
│   Roboflow      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RF-DETR Train  │
│  (COCO format)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  checkpoint.pth │
│  + class_mapping│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Export ONNX    │
│  (optionnel)    │
└─────────────────┘
```

---

## 🐛 Troubleshooting

### Erreurs courantes

<details>
<summary>❌ "PYTORCH_ENABLE_MPS_FALLBACK" sur Mac</summary>

Le script l'active automatiquement. Si vous avez encore des erreurs :

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python train_rfdetr.py ...
```
</details>

<details>
<summary>❌ "CUDA out of memory"</summary>

Réduisez la taille du batch :

```bash
python train_rfdetr.py --batch 2 --grad-accum 8 --model-size nano
```
</details>

<details>
<summary>❌ "Dataset non trouvé"</summary>

Vérifiez votre fichier CSV :
1. Les colonnes doivent être : `name,workspace,project,version`
2. Pas d'espaces après les virgules
3. Le workspace/project doit correspondre à l'URL Roboflow

```bash
# Vérifier les datasets chargés
python train_rfdetr.py --list-datasets
```
</details>

<details>
<summary>❌ "Clé API invalide"</summary>

1. Vérifiez votre clé sur [Roboflow Settings](https://app.roboflow.com/settings/api)
2. Utilisez la clé **privée**, pas la clé publique
3. Vérifiez qu'il n'y a pas d'espaces :

```bash
# Correct
--api-key abc123def456

# Incorrect
--api-key "abc123def456 "
```
</details>

<details>
<summary>⚠️ "Model is not optimized for inference"</summary>

C'est un warning, pas une erreur. Pour l'optimisation :

```python
model.optimize_for_inference()
```

Le script de prédiction peut l'ajouter automatiquement.
</details>

<details>
<summary>⚠️ "torch.meshgrid" warning</summary>

Warning PyTorch inoffensif, peut être ignoré. Pour le supprimer :

```python
import warnings
warnings.filterwarnings('ignore', message='.*torch.meshgrid.*')
```
</details>

### Performances

| Problème | Solution |
|----------|----------|
| Entraînement lent sur Mac | Normal, MPS est 5-10x plus lent que CUDA |
| GPU non utilisé à 100% | Augmentez `--batch` ou `--workers` |
| Prédictions lentes | Appelez `model.optimize_for_inference()` |

---

## 📚 Ressources

- [RF-DETR Documentation](https://roboflow.github.io/rf-detr/)
- [Roboflow API](https://docs.roboflow.com/)
- [Format COCO](https://cocodataset.org/#format-data)
- [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html)

---

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE)

---

## 🤝 Contribution

Les contributions sont les bienvenues ! 

1. Fork le projet
2. Créez une branche (`git checkout -b feature/amelioration`)
3. Committez (`git commit -m 'Ajout fonctionnalité'`)
4. Push (`git push origin feature/amelioration`)
5. Ouvrez une Pull Request
```

---

## 📋 Résumé des fichiers à créer

| Fichier | Description |
|---------|-------------|
| `train_rfdetr.py` | Script principal d'entraînement |
| `datasets.csv` | Configuration des datasets Roboflow |
| `README.md` | Documentation complète |

Voulez-vous que j'ajoute d'autres sections au README ou que je crée des fichiers supplémentaires (ex: `requirements.txt`, `Makefile`, etc.) ?
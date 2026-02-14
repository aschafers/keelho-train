"""
Script d'entraînement RF-DETR pour la détection multi-classes
Compatible: RunPod (CUDA), Mac M4 Max (MPS), CPU

Auteur: Assistant IA
Date: 2024

Usage:
    # Sur Mac M4 Max (test rapide)
    python train_rfdetr.py --api-key VOTRE_CLE --device mps --epochs 1
    
    # Sur RunPod (GPU NVIDIA)
    python train_rfdetr.py --api-key VOTRE_CLE --device cuda
    
    # Avec un fichier CSV custom
    python train_rfdetr.py --api-key VOTRE_CLE --datasets-csv mes_datasets.csv
"""

import os
from pathlib import Path
import json
import csv
import torch
import platform

# ============================================================================
# 1. INSTALLATION DES DÉPENDANCES
# ============================================================================
# pip install rfdetr roboflow torch torchvision supervision

from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRLarge
from roboflow import Roboflow


# ============================================================================
# 2. CONFIGURATION
# ============================================================================

class Config:
    """Configuration de l'entraînement RF-DETR"""
    
    # Clé API Roboflow
    ROBOFLOW_API_KEY = ""
    
    # Dossiers
    BASE_DIR = Path("./projet_detection_rfdetr")
    DATASETS_DIR = BASE_DIR / "datasets"
    MODELS_DIR = BASE_DIR / "models"
    
    # Fichier CSV des datasets (par défaut)
    DATASETS_CSV = Path("datasets.csv")
    
    # Paramètres par défaut (ajustés selon le device)
    MODEL_SIZE = "base"
    EPOCHS = 100
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 1
    RESOLUTION = 560        # RF-DETR utilise 'resolution' pas 'imgsz'
    DEVICE = "auto"
    LR = 1e-4
    NUM_WORKERS = 4
    
    # Datasets chargés depuis CSV (dictionnaire)
    DATASETS = {}


# ============================================================================
# 3. CHARGEMENT DES DATASETS DEPUIS CSV
# ============================================================================

def load_datasets_from_csv(csv_path: Path) -> dict:
    """
    Charge les datasets depuis un fichier CSV
    
    Format attendu:
        name,workspace,project,version
        sanglier,corbin-helms-wghhy,wild-boar-detection-0igwx,1
        frelon,use-case-asian-hornet-detection,asian-hornet-video-detection,1
    
    Args:
        csv_path: Chemin vers le fichier CSV
        
    Returns:
        Dictionnaire {name: {workspace, project, version}}
    """
    datasets = {}
    
    if not csv_path.exists():
        print(f"❌ Fichier CSV non trouvé: {csv_path}")
        print(f"   Créez un fichier avec le format:")
        print(f"   name,workspace,project,version")
        print(f"   sanglier,corbin-helms-wghhy,wild-boar-detection-0igwx,1")
        raise FileNotFoundError(f"Fichier CSV non trouvé: {csv_path}")
    
    print(f"\n📄 Chargement des datasets depuis: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Vérifier les colonnes requises
        required_columns = {'name', 'workspace', 'project', 'version'}
        if not required_columns.issubset(set(reader.fieldnames or [])):
            missing = required_columns - set(reader.fieldnames or [])
            raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")
        
        for row in reader:
            name = row['name'].strip()
            datasets[name] = {
                'workspace': row['workspace'].strip(),
                'project': row['project'].strip(),
                'version': int(row['version'])
            }
            print(f"   ✅ {name}: {row['workspace']}/{row['project']} v{row['version']}")
    
    if not datasets:
        raise ValueError(f"Aucun dataset trouvé dans {csv_path}")
    
    print(f"   → {len(datasets)} dataset(s) chargé(s)\n")
    
    return datasets


def save_class_mapping(output_dir: Path, datasets: dict, class_info: dict):
    """
    Sauvegarde le mapping class_id → dataset/classe dans un fichier JSON
    
    Args:
        output_dir: Dossier du modèle
        datasets: Dictionnaire des datasets utilisés
        class_info: Informations sur les classes {class_id: {name, dataset, ...}}
    """
    mapping_file = output_dir / "class_mapping.json"
    
    mapping = {
        "created_at": str(Path("date")),  # Sera remplacé par datetime
        "datasets_used": list(datasets.keys()),
        "classes": class_info,
        "total_classes": len(class_info)
    }
    
    # Ajouter la date
    from datetime import datetime
    mapping["created_at"] = datetime.now().isoformat()
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Mapping des classes sauvegardé: {mapping_file}")
    
    return mapping_file


def extract_classes_from_dataset(dataset_path: Path) -> list:
    """
    Extrait les classes depuis le fichier _annotations.coco.json
    
    Returns:
        Liste des catégories [{id, name, supercategory}, ...]
    """
    categories = []
    
    for split in ["train", "valid", "test"]:
        ann_file = dataset_path / split / "_annotations.coco.json"
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                coco = json.load(f)
            categories = coco.get("categories", [])
            break
    
    return categories


# ============================================================================
# 4. DÉTECTION ET CONFIGURATION DU DEVICE
# ============================================================================

def get_available_device() -> str:
    """Détecte le meilleur device disponible"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def detect_device_and_optimize(requested_device: str = "auto"):
    """Détecte le device et ajuste les paramètres"""
    
    print(f"\n{'='*60}")
    print(f"🖥️  DÉTECTION DU MATÉRIEL")
    print(f"{'='*60}")
    
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {platform.python_version()}")
    print(f"   PyTorch: {torch.__version__}")
    
    # Déterminer le device
    if requested_device == "auto":
        device = get_available_device()
        print(f"   Device demandé: auto → {device}")
    elif requested_device in ["0", "1", "2", "3"]:
        device = f"cuda:{requested_device}"
        print(f"   Device demandé: GPU {requested_device}")
    else:
        device = requested_device
        print(f"   Device demandé: {device}")
    
    Config.DEVICE = device
    
    # Configuration selon le device
    if device.startswith("cuda"):
        _configure_cuda(device)
    elif device == "mps":
        _configure_mps()
    else:
        _configure_cpu()
    
    # Afficher la configuration
    print(f"\n📊 Configuration finale:")
    print(f"   {'─'*40}")
    print(f"   Device:           {Config.DEVICE}")
    print(f"   Modèle:           RF-DETR-{Config.MODEL_SIZE.capitalize()}")
    print(f"   Batch size:       {Config.BATCH_SIZE}")
    print(f"   Grad accum:       {Config.GRAD_ACCUM_STEPS}")
    print(f"   Resolution:       {Config.RESOLUTION}")
    print(f"   Learning rate:    {Config.LR:.2e}")
    print(f"   Num workers:      {Config.NUM_WORKERS}")
    print(f"   {'─'*40}")


def _configure_cuda(device: str):
    """Configuration pour GPU NVIDIA"""
    gpu_id = 0 if device == "cuda" else int(device.split(":")[1])
    gpu_name = torch.cuda.get_device_name(gpu_id)
    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
    
    print(f"\n   🟢 CUDA DISPONIBLE")
    print(f"   GPU: {gpu_name}")
    print(f"   Mémoire: {gpu_memory:.1f} Go")
    
    if "A100" in gpu_name:
        Config.BATCH_SIZE = 16 if gpu_memory > 70 else 8
        Config.GRAD_ACCUM_STEPS = 1 if gpu_memory > 70 else 2
        Config.MODEL_SIZE = "large"
    elif "H100" in gpu_name:
        Config.BATCH_SIZE = 16
        Config.GRAD_ACCUM_STEPS = 1
        Config.MODEL_SIZE = "large"
    elif "4090" in gpu_name:
        Config.BATCH_SIZE = 8
        Config.GRAD_ACCUM_STEPS = 2
        Config.MODEL_SIZE = "base"
    elif "3090" in gpu_name:
        Config.BATCH_SIZE = 4
        Config.GRAD_ACCUM_STEPS = 4
        Config.MODEL_SIZE = "base"
    else:
        if gpu_memory > 20:
            Config.BATCH_SIZE = 8
            Config.GRAD_ACCUM_STEPS = 2
            Config.MODEL_SIZE = "base"
        elif gpu_memory > 10:
            Config.BATCH_SIZE = 4
            Config.GRAD_ACCUM_STEPS = 4
            Config.MODEL_SIZE = "small"
        else:
            Config.BATCH_SIZE = 2
            Config.GRAD_ACCUM_STEPS = 8
            Config.MODEL_SIZE = "nano"
    
    Config.RESOLUTION = 560
    Config.NUM_WORKERS = 8


def _configure_mps():
    """Configuration pour Mac Apple Silicon"""
    print(f"\n   🍎 APPLE SILICON (MPS)")
    print(f"   Chip: {platform.processor()}")
    
    # Activer le fallback CPU pour les opérations non supportées
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print(f"   ⚠️  PYTORCH_ENABLE_MPS_FALLBACK=1 (ops non supportées → CPU)")
    print(f"   → Configuration Mac (test/développement)")
    
    Config.MODEL_SIZE = "nano"
    Config.BATCH_SIZE = 4
    Config.GRAD_ACCUM_STEPS = 4
    Config.RESOLUTION = 640
    Config.NUM_WORKERS = 0
    Config.LR = 1e-4


def _configure_cpu():
    """Configuration pour CPU"""
    print(f"\n   🔵 CPU")
    print(f"   → Configuration CPU (très lent, debug uniquement)")
    
    Config.MODEL_SIZE = "nano"
    Config.BATCH_SIZE = 2
    Config.GRAD_ACCUM_STEPS = 8
    Config.RESOLUTION = 480
    Config.NUM_WORKERS = 0
    Config.LR = 1e-4


# ============================================================================
# 5. FONCTIONS UTILITAIRES
# ============================================================================

def setup_directories():
    """Crée les dossiers nécessaires"""
    Config.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Dossiers créés dans {Config.BASE_DIR}")


def get_model(model_size: str = None):
    """Charge le modèle RF-DETR"""
    model_size = model_size or Config.MODEL_SIZE
    
    models = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "base": RFDETRBase,
        "large": RFDETRLarge,
    }
    
    if model_size not in models:
        print(f"⚠️ Taille '{model_size}' non reconnue, utilisation de 'nano'")
        model_size = "nano"
    
    print(f"\n📦 Chargement du modèle RF-DETR-{model_size.capitalize()}...")
    model = models[model_size]()
    print(f"✅ Modèle RF-DETR-{model_size.capitalize()} chargé")
    
    return model


def download_dataset_coco(name: str, api_key: str) -> Path:
    """Télécharge un dataset depuis Roboflow au format COCO"""
    if name not in Config.DATASETS:
        available = list(Config.DATASETS.keys())
        raise ValueError(f"Dataset inconnu: {name}. Disponibles: {available}")
    
    dataset_info = Config.DATASETS[name]
    dataset_path = Config.DATASETS_DIR / name
    
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"📁 Dataset '{name}' déjà présent dans {dataset_path}")
        show_dataset_stats(dataset_path)
        return dataset_path
    
    print(f"\n📥 Téléchargement du dataset '{name}' au format COCO...")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(dataset_info["workspace"]).project(dataset_info["project"])
    
    dataset = project.version(dataset_info["version"]).download(
        model_format="coco",
        location=str(dataset_path)
    )
    
    print(f"✅ Dataset '{name}' téléchargé")
    show_dataset_stats(dataset_path)
    
    return dataset_path


def show_dataset_stats(dataset_path: Path):
    """Affiche les statistiques du dataset"""
    print(f"\n📊 Statistiques du dataset:")
    
    categories = []
    for split in ["train", "valid", "test"]:
        split_dir = dataset_path / split
        if split_dir.exists():
            images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            
            ann_file = split_dir / "_annotations.coco.json"
            if ann_file.exists():
                with open(ann_file, 'r') as f:
                    coco = json.load(f)
                num_annotations = len(coco.get("annotations", []))
                categories = [c["name"] for c in coco.get("categories", [])]
            else:
                num_annotations = "?"
            
            print(f"   {split:6s}: {len(images):5d} images, {num_annotations:5} annotations")
    
    if categories:
        print(f"   Classes: {categories}")


# ============================================================================
# 6. ENTRAÎNEMENT RF-DETR
# ============================================================================

def train_rfdetr(dataset_name: str, epochs: int = None) -> str:
    """
    Entraîne RF-DETR sur un dataset
    
    Args:
        dataset_name: Nom du dataset
        epochs: Nombre d'époques
        
    Returns:
        Chemin vers le modèle entraîné
    """
    epochs = epochs or Config.EPOCHS
    
    print(f"\n{'='*70}")
    print(f"🚀 ENTRAÎNEMENT RF-DETR")
    print(f"{'='*70}")
    
    # 1. Télécharger le dataset
    dataset_path = download_dataset_coco(dataset_name, Config.ROBOFLOW_API_KEY)
    
    # 2. Extraire les classes du dataset
    categories = extract_classes_from_dataset(dataset_path)
    print(f"\n📋 Classes détectées:")
    class_info = {}
    for cat in categories:
        class_info[cat['id']] = {
            'name': cat['name'],
            'dataset': dataset_name,
            'supercategory': cat.get('supercategory', '')
        }
        print(f"   ID {cat['id']}: {cat['name']}")
    
    # 3. Charger le modèle
    model = get_model(Config.MODEL_SIZE)
    
    # 4. Chemin de sortie
    output_dir = Config.MODELS_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. Sauvegarder le mapping des classes
    save_class_mapping(output_dir, {dataset_name: Config.DATASETS[dataset_name]}, class_info)
    
    # 6. Afficher la configuration
    print(f"\n{'='*70}")
    print(f"📊 CONFIGURATION D'ENTRAÎNEMENT")
    print(f"{'='*70}")
    print(f"   Device:           {Config.DEVICE}")
    print(f"   Modèle:           RF-DETR-{Config.MODEL_SIZE.capitalize()}")
    print(f"   Époques:          {epochs}")
    print(f"   Batch size:       {Config.BATCH_SIZE}")
    print(f"   Grad accum:       {Config.GRAD_ACCUM_STEPS}")
    print(f"   Resolution:       {Config.RESOLUTION}")
    print(f"   Learning rate:    {Config.LR:.2e}")
    print(f"   Num workers:      {Config.NUM_WORKERS}")
    print(f"   Dataset dir:      {dataset_path}")
    print(f"   Output dir:       {output_dir}")
    print(f"   Classes:          {len(class_info)}")
    print(f"{'='*70}\n")
    
    # 7. Estimer le temps
    if Config.DEVICE.startswith("cuda"):
        time_estimate = f"~{epochs * 2} minutes"
    elif Config.DEVICE == "mps":
        time_estimate = f"~{epochs * 10} minutes"
    else:
        time_estimate = f"~{epochs * 60} minutes (CPU très lent!)"
    
    print(f"🏋️ Début de l'entraînement...")
    print(f"   Temps estimé: {time_estimate}\n")
    
    # 8. Lancer l'entraînement
    try:
        model.train(
            dataset_dir=str(dataset_path),
            epochs=epochs,
            batch_size=Config.BATCH_SIZE,
            grad_accum_steps=Config.GRAD_ACCUM_STEPS,
            lr=Config.LR,
            output_dir=str(output_dir),
            resolution=Config.RESOLUTION,
            device=Config.DEVICE,
            num_workers=Config.NUM_WORKERS,
        )
    except TypeError as e:
        print(f"⚠️ Certains paramètres non supportés, utilisation config minimale")
        print(f"   Erreur: {e}\n")
        model.train(
            dataset_dir=str(dataset_path),
            epochs=epochs,
            batch_size=Config.BATCH_SIZE,
            grad_accum_steps=Config.GRAD_ACCUM_STEPS,
            lr=Config.LR,
            output_dir=str(output_dir),
        )
    
    # 9. Trouver le meilleur modèle
    model_path = find_best_model(output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ ENTRAÎNEMENT TERMINÉ!")
    print(f"{'='*70}")
    print(f"📦 Modèle sauvegardé: {model_path}")
    print(f"📋 Mapping classes:   {output_dir / 'class_mapping.json'}")
    
    return str(model_path)


def find_best_model(output_dir: Path) -> Path:
    """Trouve le meilleur modèle dans le dossier de sortie"""
    
    possible_names = [
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
        "checkpoint.pth",
        "best_model.pt",
        "best.pt",
    ]
    
    for name in possible_names:
        path = output_dir / name
        if path.exists():
            return path
    
    for ext in ["*.pth", "*.pt"]:
        files = list(output_dir.glob(ext))
        if files:
            return max(files, key=lambda p: p.stat().st_mtime)
    
    for ext in ["**/*.pth", "**/*.pt"]:
        files = list(output_dir.glob(ext))
        if files:
            return max(files, key=lambda p: p.stat().st_mtime)
    
    raise FileNotFoundError(f"Aucun modèle trouvé dans {output_dir}")


# ============================================================================
# 7. ÉVALUATION ET PRÉDICTION
# ============================================================================

def predict_test(model_path: str = None, image_path: str = None, save_dir: str = None):
    """Test de prédiction rapide"""
    from PIL import Image
    import requests
    from io import BytesIO
    
    print(f"\n{'='*70}")
    print(f"🔍 TEST DE PRÉDICTION")
    print(f"{'='*70}")
    
    model = get_model(Config.MODEL_SIZE)
    
    # Charger le mapping des classes si disponible
    class_names = {}
    if model_path and Path(model_path).exists():
        mapping_file = Path(model_path).parent / "class_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
            for class_id, info in mapping.get('classes', {}).items():
                class_names[int(class_id)] = info['name']
            print(f"   📋 Classes chargées: {class_names}")
        
        print(f"   Chargement des poids: {model_path}")
        model = type(model)(pretrain_weights=str(model_path))
        print(f"   ✅ Modèle custom chargé")
    else:
        print(f"   ℹ️ Utilisation du modèle pré-entraîné COCO")
    
    # Image de test
    if image_path:
        if image_path.startswith("http"):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        print(f"   📷 Image: {image_path}")
    else:
        test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Wild_boar.jpg/1280px-Wild_boar.jpg"
        response = requests.get(test_url)
        image = Image.open(BytesIO(response.content))
        print(f"   📷 Image de test (sanglier Wikipedia)")
    
    print(f"   📐 Taille: {image.size}")
    
    # Prédiction
    detections = model.predict(image, threshold=0.5)
    
    print(f"\n   📦 {len(detections)} détection(s):")
    
    if len(detections) > 0:
        if hasattr(detections, 'class_id'):
            for i in range(len(detections.class_id)):
                class_id = detections.class_id[i]
                conf = detections.confidence[i]
                class_name = class_names.get(class_id, f"classe_{class_id}")
                print(f"      {i+1}. {class_name} (ID:{class_id}): {conf:.1%}")
    
    return detections


def export_for_jetson(model_path: str, formats: list = None):
    """Exporte le modèle pour Jetson"""
    formats = formats or ["onnx"]
    
    print(f"\n{'='*70}")
    print(f"📤 EXPORT POUR JETSON ORIN")
    print(f"{'='*70}")
    print(f"   Modèle: {model_path}")
    
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"   ❌ Modèle non trouvé")
        return
    
    model = get_model(Config.MODEL_SIZE)
    
    output_dir = model_file.parent / "exports"
    output_dir.mkdir(exist_ok=True)
    
    for fmt in formats:
        try:
            if fmt == "onnx":
                export_path = output_dir / f"{model_file.stem}.onnx"
                model.export(str(export_path))
                print(f"   ✅ ONNX: {export_path}")
        except Exception as e:
            print(f"   ❌ Erreur export {fmt}: {e}")


# ============================================================================
# 8. FONCTIONS PRINCIPALES
# ============================================================================

def train_single_model(dataset_name: str):
    """Pipeline complet pour un dataset"""
    setup_directories()
    
    model_path = train_rfdetr(dataset_name)
    
    try:
        predict_test(model_path)
    except Exception as e:
        print(f"⚠️ Erreur prédiction test: {e}")
    
    if Config.EPOCHS > 5:
        try:
            export_for_jetson(model_path, ["onnx"])
        except Exception as e:
            print(f"⚠️ Erreur export: {e}")
    
    return model_path


def train_all_models():
    """Entraîne tous les datasets du CSV"""
    models = {}
    setup_directories()
    
    for dataset_name in Config.DATASETS.keys():
        try:
            print(f"\n\n{'#'*70}")
            print(f"# DATASET: {dataset_name.upper()}")
            print(f"{'#'*70}")
            
            model_path = train_rfdetr(dataset_name)
            models[dataset_name] = model_path
            
        except Exception as e:
            print(f"❌ Erreur pour {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"📋 RÉSUMÉ")
    print(f"{'='*70}")
    for name, path in models.items():
        print(f"   ✅ {name}: {path}")
    
    return models


# ============================================================================
# 9. POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Entraînement RF-DETR - Compatible CUDA, MPS (Mac), CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Sur Mac M4 Max (test rapide 1 epoch)
  python train_rfdetr.py --api-key CLE --device mps --epochs 1
  
  # Sur RunPod avec GPU NVIDIA
  python train_rfdetr.py --api-key CLE --device cuda
  
  # Avec fichier CSV custom
  python train_rfdetr.py --api-key CLE --datasets-csv mes_datasets.csv
  
  # Entraîner un dataset spécifique
  python train_rfdetr.py --api-key CLE --dataset sanglier

Format du fichier CSV:
  name,workspace,project,version
  sanglier,corbin-helms-wghhy,wild-boar-detection-0igwx,1
  frelon,use-case-asian-hornet-detection,asian-hornet-video-detection,1
        """
    )
    
    # Arguments principaux
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=None,
        help="Dataset spécifique à entraîner (ou 'all' pour tous)"
    )
    parser.add_argument(
        "--datasets-csv",
        type=str,
        default="datasets.csv",
        help="Chemin vers le fichier CSV des datasets (défaut: datasets.csv)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=False,
        help="Clé API Roboflow"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'cuda:0', 'mps', 'cpu' (défaut: auto)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Nombre d'époques"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Taille du batch"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["nano", "small", "base", "large"],
        default=None,
        help="Taille du modèle"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Résolution des images"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Nombre de workers"
    )
    
    # Modes spéciaux
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Mode prédiction"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Chemin vers le modèle (pour prédiction)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Image pour prédiction"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Liste les datasets disponibles dans le CSV"
    )

    args = parser.parse_args()
    
    # Charger les datasets depuis le CSV
    Config.DATASETS_CSV = Path(args.datasets_csv)
    
    try:
        Config.DATASETS = load_datasets_from_csv(Config.DATASETS_CSV)
    except FileNotFoundError:
        if not args.list_datasets:
            print(f"\n💡 Créez le fichier {args.datasets_csv} avec le format:")
            print(f"   name,workspace,project,version")
            print(f"   sanglier,corbin-helms-wghhy,wild-boar-detection-0igwx,1")
            exit(1)
    
    # Mode liste des datasets
    if args.list_datasets:
        print(f"\n📋 Datasets disponibles dans {Config.DATASETS_CSV}:")
        for name, info in Config.DATASETS.items():
            print(f"   • {name}: {info['workspace']}/{info['project']} v{info['version']}")
        exit(0)
    
    # Détection du device et configuration auto
    detect_device_and_optimize(args.device)
    
    # Override des paramètres si spécifiés
    if args.api_key:
        Config.ROBOFLOW_API_KEY = args.api_key
    
    if args.epochs is not None:
        Config.EPOCHS = args.epochs
    
    if args.batch is not None:
        Config.BATCH_SIZE = args.batch
    
    if args.model_size is not None:
        Config.MODEL_SIZE = args.model_size
    
    if args.lr is not None:
        Config.LR = args.lr
    
    if args.resolution is not None:
        Config.RESOLUTION = args.resolution
    
    if args.grad_accum is not None:
        Config.GRAD_ACCUM_STEPS = args.grad_accum
    
    if args.workers is not None:
        Config.NUM_WORKERS = args.workers
    
    # Mode prédiction
    if args.predict:
        setup_directories()
        predict_test(args.model, args.image)
        exit(0)
    
    # Vérifier la clé API
    if not Config.ROBOFLOW_API_KEY:
        Config.ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
    
    if not Config.ROBOFLOW_API_KEY:
        print("❌ Clé API Roboflow requise")
        print("   --api-key VOTRE_CLE")
        print("   ou: export ROBOFLOW_API_KEY=VOTRE_CLE")
        exit(1)
    
    # Déterminer le dataset à entraîner
    if args.dataset is None:
        # Si un seul dataset dans le CSV, l'utiliser
        if len(Config.DATASETS) == 1:
            args.dataset = list(Config.DATASETS.keys())[0]
            print(f"ℹ️ Un seul dataset dans le CSV, utilisation de: {args.dataset}")
        else:
            print(f"❌ Plusieurs datasets disponibles, spécifiez --dataset:")
            for name in Config.DATASETS.keys():
                print(f"   • {name}")
            print(f"   • all (pour tous les entraîner)")
            exit(1)
    
    # Vérifier que le dataset existe
    if args.dataset != "all" and args.dataset not in Config.DATASETS:
        print(f"❌ Dataset '{args.dataset}' non trouvé dans {Config.DATASETS_CSV}")
        print(f"   Disponibles: {list(Config.DATASETS.keys())}")
        exit(1)
    
    # Bannière
    device_emoji = "🖥️" if Config.DEVICE.startswith("cuda") else "🍎" if Config.DEVICE == "mps" else "💻"
    
    print("\n")
    print("🐗" * 35)
    print("🐗" + " " * 66 + "🐗")
    print("🐗" + "   RF-DETR TRAINING".center(66) + "🐗")
    print("🐗" + f"   {device_emoji} Device: {Config.DEVICE}".center(66) + "🐗")
    print("🐗" + f"   📄 CSV: {Config.DATASETS_CSV}".center(66) + "🐗")
    print("🐗" + " " * 66 + "🐗")
    print("🐗" * 35)
    print("\n")
    
    # Lancer l'entraînement
    if args.dataset == "all":
        train_all_models()
    else:
        train_single_model(args.dataset)
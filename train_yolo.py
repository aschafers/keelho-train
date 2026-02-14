"""
Script d'entraînement YOLO pour la détection de sangliers et frelons asiatiques
Auteur: Assistant IA
Date: 2024
"""

import os
from pathlib import Path

# ============================================================================
# 1. INSTALLATION DES DÉPENDANCES
# ============================================================================
# Exécuter ces commandes dans le terminal avant de lancer le script :
# pip install ultralytics roboflow torch torchvision

from ultralytics import YOLO
from roboflow import Roboflow


# ============================================================================
# 2. CONFIGURATION
# ============================================================================

class Config:
    """Configuration de l'entraînement"""
    
    # Clé API Roboflow (à obtenir sur https://app.roboflow.com/settings/api)
    ROBOFLOW_API_KEY = ""  # ⚠️ À REMPLACER
    
    # Dossier de base pour les datasets
    BASE_DIR = Path("./projet_detection")
    DATASETS_DIR = BASE_DIR / "datasets"
    
    # Paramètres d'entraînement
    MODEL_SIZE = "yolo11n.pt"  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    EPOCHS = 100               # Nombre d'époques
    BATCH_SIZE = 16            # Taille du batch (réduire si mémoire insuffisante)
    IMAGE_SIZE = 640           # Taille des images
    DEVICE = "0"               # "0" pour GPU, "cpu" pour CPU
    
    # Datasets Roboflow
    DATASETS = {
        "sanglier": {
            "workspace": "corbin-helms-wghhy",
            "project": "wild-boar-detection-0igwx",
            "version": 1
        },
        "frelon": {
            "workspace": "use-case-asian-hornet-detection",
            "project": "asian-hornet-video-detection",
            "version": 1
        }
    }


# ============================================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================================

def setup_directories():
    """Crée les dossiers nécessaires"""
    Config.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Dossiers créés dans {Config.BASE_DIR}")


def download_dataset(name: str, api_key: str) -> Path:
    """
    Télécharge un dataset depuis Roboflow
    
    Args:
        name: Nom du dataset ("sanglier" ou "frelon")
        api_key: Clé API Roboflow
        
    Returns:
        Chemin vers le dataset téléchargé
    """
    if name not in Config.DATASETS:
        raise ValueError(f"Dataset inconnu: {name}")
    
    dataset_info = Config.DATASETS[name]
    dataset_path = Config.DATASETS_DIR / name
    
    # Vérifier si déjà téléchargé
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"📁 Dataset '{name}' déjà présent dans {dataset_path}")
        return dataset_path
    
    print(f"📥 Téléchargement du dataset '{name}'...")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(dataset_info["workspace"]).project(dataset_info["project"])
    dataset = project.version(dataset_info["version"]).download(
        model_format="yolov8",
        location=str(dataset_path)
    )
    
    print(f"✅ Dataset '{name}' téléchargé dans {dataset_path}")
    return dataset_path


def find_data_yaml(dataset_path: Path) -> Path:
    """Trouve le fichier data.yaml dans le dataset"""
    # Chercher data.yaml dans le dossier et sous-dossiers
    for yaml_file in dataset_path.rglob("data.yaml"):
        return yaml_file
    
    # Alternatives possibles
    for alt_name in ["dataset.yaml", "config.yaml"]:
        for yaml_file in dataset_path.rglob(alt_name):
            return yaml_file
    
    raise FileNotFoundError(f"Aucun fichier data.yaml trouvé dans {dataset_path}")


def train_model(dataset_name: str, data_yaml_path: Path, epochs: int = None) -> str:
    """
    Entraîne un modèle YOLO sur un dataset
    
    Args:
        dataset_name: Nom du dataset pour le nom du projet
        data_yaml_path: Chemin vers le fichier data.yaml
        epochs: Nombre d'époques (utilise Config.EPOCHS par défaut)
        
    Returns:
        Chemin vers le meilleur modèle entraîné
    """
    epochs = epochs or Config.EPOCHS
    
    print(f"\n{'='*60}")
    print(f"🚀 Début de l'entraînement pour '{dataset_name}'")
    print(f"{'='*60}")
    print(f"📊 Configuration:")
    print(f"   - Modèle de base: {Config.MODEL_SIZE}")
    print(f"   - Époques: {epochs}")
    print(f"   - Batch size: {Config.BATCH_SIZE}")
    print(f"   - Taille image: {Config.IMAGE_SIZE}")
    print(f"   - Device: {Config.DEVICE}")
    print(f"   - Data YAML: {data_yaml_path}")
    print(f"{'='*60}\n")
    
    # Créer explicitement tous les dossiers nécessaires
    weights_dir = Config.BASE_DIR / "runs" / dataset_name / "train" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Dossiers créés: {weights_dir}")
    
    # Charger le modèle pré-entraîné
    model = YOLO(Config.MODEL_SIZE)
    
    # Lancer l'entraînement
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        batch=Config.BATCH_SIZE,
        imgsz=Config.IMAGE_SIZE,
        device=Config.DEVICE,
        project=str(Config.BASE_DIR / "runs" / dataset_name),
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        patience=20,        # Early stopping après 20 époques sans amélioration
        save=True,
        save_period=10,     # Sauvegarder tous les 10 époques
        plots=True,         # Générer les graphiques
        val=True,           # Validation pendant l'entraînement
    )
    
    # YOLO sauvegarde dans runs/detect/{project}/{name}/weights/
    # Le chemin réel peut différer de celui configuré
    weights_dir = Path("runs/detect") / Config.BASE_DIR / "runs" / dataset_name / "train" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Dossiers créés: {weights_dir}")
    
    # Chercher best.pt dans les différents emplacements possibles
    possible_paths = [
        weights_dir / "best.pt",  # runs/detect/projet_detection/runs/sanglier/train/weights/best.pt
        Path("runs/detect") / Config.BASE_DIR / "runs" / dataset_name / "train" / "weights" / "best.pt",
        Path.cwd() / "runs" / "detect" / str(Config.BASE_DIR) / "runs" / dataset_name / "train" / "weights" / "best.pt",
    ]
    
    # Chercher aussi dans tout le dossier runs/detect
    runs_detect_dir = Path("runs/detect")
    if runs_detect_dir.exists():
        for pt_file in runs_detect_dir.rglob(f"{dataset_name}/**/weights/best.pt"):
            possible_paths.append(pt_file)
    
    best_model_path = None
    for path in possible_paths:
        if path.exists():
            best_model_path = path
            break
    
    if best_model_path:
        print(f"\n✅ Entraînement terminé!")
        print(f"📦 Meilleur modèle sauvegardé: {best_model_path}")
        return str(best_model_path)
    else:
        # Chercher last.pt comme fallback
        for path in possible_paths:
            last_path = path.parent / "last.pt"
            if last_path.exists():
                print(f"\n⚠️  best.pt non trouvé, mais last.pt existe")
                print(f"📦 Utilisation de last.pt: {last_path}")
                return str(last_path)
        
        # Chercher tout fichier .pt
        if runs_detect_dir.exists():
            pt_files = list(runs_detect_dir.rglob("*.pt"))
            if pt_files:
                print(f"\n⚠️  best.pt non trouvé, utilisation de: {pt_files[0]}")
                return str(pt_files[0])
        
        raise FileNotFoundError(
            f"Aucun fichier modèle (.pt) trouvé dans runs/detect/\n"
            f"L'entraînement a peut-être échoué. Vérifiez les logs ci-dessus."
        )


def evaluate_model(model_path: str, data_yaml_path: Path):
    """
    Évalue un modèle entraîné sur le set de validation
    
    Args:
        model_path: Chemin vers le modèle .pt
        data_yaml_path: Chemin vers le fichier data.yaml
    """
    print(f"\n📊 Évaluation du modèle: {model_path}")
    
    # Vérifier que le fichier modèle existe
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Le fichier modèle n'existe pas: {model_path}\n"
            f"L'entraînement a peut-être échoué ou le modèle n'a pas été sauvegardé."
        )
    
    print(f"   ✅ Fichier modèle trouvé: {model_file}")
    
    model = YOLO(model_path)
    results = model.val(data=str(data_yaml_path))
    
    print("\n📈 Résultats de l'évaluation:")
    print(f"   - mAP50: {results.box.map50:.4f}")
    print(f"   - mAP50-95: {results.box.map:.4f}")
    print(f"   - Précision: {results.box.mp:.4f}")
    print(f"   - Rappel: {results.box.mr:.4f}")
    
    return results


def export_model(model_path: str, formats: list = None):
    """
    Exporte le modèle vers différents formats
    
    Args:
        model_path: Chemin vers le modèle .pt
        formats: Liste des formats d'export (défaut: ["onnx"])
    """
    formats = formats or ["onnx"]
    
    print(f"\n📤 Export du modèle vers: {formats}")
    
    # Vérifier que le fichier modèle existe
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"   ❌ Erreur: Le fichier modèle n'existe pas: {model_path}")
        print(f"   ⏭️  Export ignoré")
        return
    
    print(f"   ✅ Fichier modèle trouvé: {model_file}")
    
    model = YOLO(model_path)
    
    for fmt in formats:
        try:
            export_path = model.export(format=fmt)
            print(f"   ✅ Export {fmt}: {export_path}")
        except Exception as e:
            print(f"   ❌ Erreur export {fmt}: {e}")


def predict_on_image(model_path: str, image_path: str, save_dir: str = None):
    """
    Effectue une prédiction sur une image
    
    Args:
        model_path: Chemin vers le modèle .pt
        image_path: Chemin vers l'image ou URL
        save_dir: Dossier de sauvegarde des résultats
    """
    print(f"\n🔍 Prédiction sur: {image_path}")
    
    # Vérifier que le fichier modèle existe
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Le fichier modèle n'existe pas: {model_path}\n"
            f"Vérifiez le chemin du modèle."
        )
    
    print(f"   ✅ Fichier modèle trouvé: {model_file}")
    
    model = YOLO(model_path)
    
    results = model.predict(
        source=image_path,
        save=True,
        save_dir=save_dir,
        conf=0.25,          # Seuil de confiance
        iou=0.45,           # Seuil IoU pour NMS
        show_labels=True,
        show_conf=True,
    )
    
    # Afficher les détections
    for result in results:
        boxes = result.boxes
        print(f"\n   📦 {len(boxes)} objet(s) détecté(s):")
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf = float(box.conf[0])
            print(f"      - {cls_name}: {conf:.2%}")
    
    return results


# ============================================================================
# 4. FONCTIONS PRINCIPALES
# ============================================================================

def train_single_model(dataset_name: str):
    """
    Entraîne un modèle sur un seul dataset
    
    Args:
        dataset_name: "sanglier" ou "frelon"
    """
    setup_directories()
    
    # Télécharger le dataset
    dataset_path = download_dataset(dataset_name, Config.ROBOFLOW_API_KEY)
    
    # Trouver le fichier data.yaml
    data_yaml = find_data_yaml(dataset_path)
    
    # Entraîner
    model_path = train_model(dataset_name, data_yaml)
    
    # Évaluer
    evaluate_model(model_path, data_yaml)
    
    # Exporter
    export_model(model_path, ["onnx"])
    
    return model_path


def train_all_models():
    """Entraîne un modèle pour chaque dataset"""
    models = {}
    
    for dataset_name in Config.DATASETS.keys():
        try:
            model_path = train_single_model(dataset_name)
            models[dataset_name] = model_path
        except Exception as e:
            print(f"❌ Erreur pour {dataset_name}: {e}")
    
    print("\n" + "="*60)
    print("📋 RÉSUMÉ DES MODÈLES ENTRAÎNÉS")
    print("="*60)
    for name, path in models.items():
        print(f"   - {name}: {path}")
    
    return models


# ============================================================================
# 5. POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraînement YOLO pour sangliers et frelons")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["sanglier", "frelon", "all"],
        default="sanglier",
        help="Dataset à entraîner (défaut: all)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Nombre d'époques (défaut: 100)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Taille du batch (défaut: 16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: '0' pour GPU, 'cpu' pour CPU (défaut: 0)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Clé API Roboflow"
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Chemin image pour prédiction (nécessite --model)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Chemin vers le modèle pour prédiction"
    )

    args = parser.parse_args()
    
    # Mettre à jour la configuration
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch
    Config.DEVICE = args.device
 
    if args.api_key:
        Config.ROBOFLOW_API_KEY = args.api_key
    
    # Mode prédiction
    if args.predict:
        if not args.model:
            print("❌ Erreur: --model requis pour la prédiction")
            exit(1)
        predict_on_image(args.model, args.predict)
        exit(0)
    
    # Vérifier la clé API
    if Config.ROBOFLOW_API_KEY == "VOTRE_CLE_API_ICI":
        print("❌ Erreur: Veuillez configurer votre clé API Roboflow")
        print("   1. Créez un compte sur https://roboflow.com")
        print("   2. Allez dans Settings > API")
        print("   3. Copiez votre clé et remplacez VOTRE_CLE_API_ICI dans le script")
        print("   Ou utilisez: --api-key VOTRE_CLE")
        exit(1)
    
    # Lancer l'entraînement
    print("\n" + "🐗🐝 "*10)
    print("   ENTRAÎNEMENT YOLO - SANGLIERS & FRELONS")
    print("🐗🐝 "*10 + "\n")
    
    if args.dataset == "all":
        train_all_models()
    else:
        train_single_model(args.dataset)
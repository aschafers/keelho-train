"""
Script d'entraînement RF-DETR pour la détection multi-classes
Compatible: RunPod (CUDA), Mac M4 Max (MPS), CPU

Modes d'entraînement:
  - single: Un modèle par dataset (défaut)
  - merged: Fusion des datasets → Un seul modèle multi-classes

Fonctionnalités:
  - Gestion interactive des classes (fusion, renommage)
  - Validation et correction automatique des category_id COCO
  - Early stopping
  - Support multi-GPU

Auteur: Assistant IA
Date: 2024
"""

import os
from pathlib import Path
import json
import csv
import shutil
import torch
import platform
import time
from datetime import datetime
import wandb
import signal
from typing import Dict, List, Tuple, Optional, Set

# ============================================================================
# 1. INSTALLATION DES DÉPENDANCES
# ============================================================================
# pip install rfdetr roboflow torch torchvision supervision

from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRLarge
from roboflow import Roboflow

def remove_empty_classes_from_dataset(dataset_path: Path) -> List[Dict]:
    """
    Supprime les classes avec 0 annotations d'un dataset COCO
    
    Cette fonction doit être appelée APRÈS le téléchargement et AVANT
    l'affichage des classes à l'utilisateur.
    
    Args:
        dataset_path: Chemin du dataset
    
    Returns:
        Liste des classes supprimées [{"id": 0, "name": "pig", "source": "..."}, ...]
    """
    removed_classes = []
    
    for split in ["train", "valid", "test"]:
        ann_file = dataset_path / split / "_annotations.coco.json"
        if not ann_file.exists():
            continue
        
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        
        categories = coco.get("categories", [])
        annotations = coco.get("annotations", [])
        
        if not categories:
            continue
        
        # Compter les annotations par catégorie
        cat_annotation_count = {cat["id"]: 0 for cat in categories}
        for ann in annotations:
            cat_id = ann["category_id"]
            if cat_id in cat_annotation_count:
                cat_annotation_count[cat_id] += 1
        
        # Identifier les catégories vides
        empty_cat_ids = {cat_id for cat_id, count in cat_annotation_count.items() if count == 0}
        
        if not empty_cat_ids:
            continue
        
        # Enregistrer les classes supprimées
        for cat in categories:
            if cat["id"] in empty_cat_ids:
                removed_classes.append({
                    "id": cat["id"],
                    "name": cat["name"],
                    "split": split
                })
        
        # Filtrer les catégories non vides
        new_categories = [cat for cat in categories if cat["id"] not in empty_cat_ids]
        
        # Créer le mapping old_id -> new_id (continu à partir de 0)
        old_to_new = {}
        for new_id, cat in enumerate(sorted(new_categories, key=lambda x: x["id"])):
            old_to_new[cat["id"]] = new_id
            cat["id"] = new_id
        
        # Mettre à jour les annotations
        for ann in annotations:
            if ann["category_id"] in old_to_new:
                ann["category_id"] = old_to_new[ann["category_id"]]
        
        # Sauvegarder
        coco["categories"] = sorted(new_categories, key=lambda x: x["id"])
        
        # Backup
        backup_file = ann_file.with_suffix('.json.backup_before_cleanup')
        if not backup_file.exists():
            shutil.copy(ann_file, backup_file)
        
        with open(ann_file, 'w') as f:
            json.dump(coco, f, indent=2)
    
    # Afficher le résumé
    if removed_classes:
        # Dédupliquer par nom (une classe peut être dans plusieurs splits)
        unique_removed = {}
        for cls in removed_classes:
            if cls["name"] not in unique_removed:
                unique_removed[cls["name"]] = cls
        
        print(f"\n   🧹 {len(unique_removed)} classe(s) vide(s) supprimée(s) automatiquement:")
        names = [f"'{name}'" for name in sorted(unique_removed.keys())]
        # Afficher sur plusieurs lignes si beaucoup
        if len(names) <= 5:
            print(f"      {', '.join(names)}")
        else:
            for i in range(0, len(names), 5):
                print(f"      {', '.join(names[i:i+5])}")
    
    return removed_classes
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
    
    # Mode d'entraînement: "single" ou "merged"
    TRAINING_MODE = "single"
    
    # Nom du modèle fusionné (si mode merged)
    MERGED_MODEL_NAME = "merged_model"
    
    # Mode interactif (True = demande confirmation pour fusion/renommage)
    INTERACTIVE_MODE = True
    
    # ========================================================================
    # PARAMÈTRES PAR DÉFAUT (PRODUCTION)
    # ========================================================================
    MODEL_SIZE = "base"
    EPOCHS = 200
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 1
    RESOLUTION = 640
    DEVICE = "auto"
    LR = 1e-4
    NUM_WORKERS = 4
    
    # Early Stopping
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 50
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Datasets chargés depuis CSV (dictionnaire)
    DATASETS = {}


# ============================================================================
# 3. CHARGEMENT DES DATASETS DEPUIS CSV
# ============================================================================

def load_datasets_from_csv(csv_path: Path) -> dict:
    """
    Charge les datasets depuis un fichier CSV
    
    Format attendu (colonnes obligatoires + optionnelles):
        name,workspace,project,version,url,notes
        sanglier,workspace1,project1,1,https://...,Mon commentaire
        
    Les colonnes 'url', 'notes' et autres sont optionnelles et stockées.
    """
    datasets = {}
    
    if not csv_path.exists():
        print(f"❌ Fichier CSV non trouvé: {csv_path}")
        print(f"   Créez un fichier avec le format:")
        print(f"   name,workspace,project,version,url")
        print(f"   sanglier,corbin-helms-wghhy,wild-boar-detection-0igwx,1,https://app.roboflow.com/...")
        raise FileNotFoundError(f"Fichier CSV non trouvé: {csv_path}")
    
    print(f"\n📄 Chargement des datasets depuis: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Vérifier les colonnes OBLIGATOIRES seulement
        required_columns = {'name', 'workspace', 'project', 'version'}
        available_columns = set(reader.fieldnames or [])
        
        if not required_columns.issubset(available_columns):
            missing = required_columns - available_columns
            raise ValueError(f"Colonnes obligatoires manquantes dans le CSV: {missing}")
        
        # Afficher les colonnes détectées
        optional_columns = available_columns - required_columns
        if optional_columns:
            print(f"   📋 Colonnes optionnelles détectées: {optional_columns}")
        
        for row in reader:
            name = row['name'].strip()
            
            # Stocker toutes les informations (obligatoires + optionnelles)
            datasets[name] = {
                'workspace': row['workspace'].strip(),
                'project': row['project'].strip(),
                'version': int(row['version']),
            }
            
            # Ajouter les colonnes optionnelles si présentes
            if 'url' in row and row['url']:
                datasets[name]['url'] = row['url'].strip()
            
            if 'notes' in row and row['notes']:
                datasets[name]['notes'] = row['notes'].strip()
            
            # Stocker toutes les autres colonnes optionnelles
            for col in optional_columns:
                if col in row and row[col] and col not in ['url', 'notes']:
                    datasets[name][col] = row[col].strip()
            
            # Affichage
            url_info = f" 🔗" if 'url' in datasets[name] else ""
            print(f"   ✅ {name}: {row['workspace']}/{row['project']} v{row['version']}{url_info}")
    
    if not datasets:
        raise ValueError(f"Aucun dataset trouvé dans {csv_path}")
    
    print(f"   → {len(datasets)} dataset(s) chargé(s)\n")
    
    return datasets


# ============================================================================
# 4. GESTION INTERACTIVE DES CLASSES
# ============================================================================

def analyze_dataset_classes(dataset_path: Path) -> Dict:
    """
    Analyse les classes d'un dataset et compte les images/annotations par classe
    
    Returns:
        {
            "categories": [{"id": 0, "name": "pig", "images": 3500, "annotations": 5800}, ...],
            "total_images": 4000,
            "total_annotations": 6315
        }
    """
    result = {
        "categories": [],
        "total_images": 0,
        "total_annotations": 0
    }
    
    # Collecter les stats de tous les splits
    all_categories = {}  # id -> {"name": ..., "images": set(), "annotations": 0}
    all_image_ids = set()
    
    for split in ["train", "valid", "test"]:
        ann_file = dataset_path / split / "_annotations.coco.json"
        if not ann_file.exists():
            continue
        
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        
        # Initialiser les catégories
        for cat in coco.get("categories", []):
            if cat["id"] not in all_categories:
                all_categories[cat["id"]] = {
                    "id": cat["id"],
                    "name": cat["name"],
                    "images": set(),
                    "annotations": 0
                }
        
        # Compter les annotations et images par catégorie
        for ann in coco.get("annotations", []):
            cat_id = ann["category_id"]
            if cat_id in all_categories:
                all_categories[cat_id]["annotations"] += 1
                # Utiliser un ID unique combinant split et image_id
                unique_img_id = f"{split}_{ann['image_id']}"
                all_categories[cat_id]["images"].add(unique_img_id)
        
        # Compter les images totales
        for img in coco.get("images", []):
            unique_img_id = f"{split}_{img['id']}"
            all_image_ids.add(unique_img_id)
        
        result["total_annotations"] += len(coco.get("annotations", []))
    
    result["total_images"] = len(all_image_ids)
    
    # Convertir les sets en counts
    for cat_id, cat_info in all_categories.items():
        result["categories"].append({
            "id": cat_info["id"],
            "name": cat_info["name"],
            "images": len(cat_info["images"]),
            "annotations": cat_info["annotations"]
        })
    
    # Trier par ID
    result["categories"].sort(key=lambda x: x["id"])
    
    return result


def display_classes_table(dataset_name: str, class_stats: Dict):
    """
    Affiche un tableau des classes avec leurs statistiques
    """
    categories = class_stats["categories"]
    
    print(f"\n{'='*70}")
    print(f"📦 Dataset: {dataset_name}")
    print(f"{'='*70}")
    
    if not categories:
        print("   ❌ Aucune classe trouvée!")
        return
    
    # Calculer les largeurs de colonnes
    max_name_len = max(len(cat["name"]) for cat in categories)
    max_name_len = max(max_name_len, 10)  # Minimum 10 caractères
    
    # En-tête
    print(f"\n   ┌{'─'*6}┬{'─'*(max_name_len+2)}┬{'─'*10}┬{'─'*14}┐")
    print(f"   │ {'ID':^4} │ {'Nom':^{max_name_len}} │ {'Images':^8} │ {'Annotations':^12} │")
    print(f"   ├{'─'*6}┼{'─'*(max_name_len+2)}┼{'─'*10}┼{'─'*14}┤")
    
    # Lignes
    for cat in categories:
        print(f"   │ {cat['id']:^4} │ {cat['name']:<{max_name_len}} │ {cat['images']:>8} │ {cat['annotations']:>12} │")
    
    # Pied
    print(f"   └{'─'*6}┴{'─'*(max_name_len+2)}┴{'─'*10}┴{'─'*14}┘")
    
    # Résumé
    print(f"\n   📊 Total: {len(categories)} classe(s), {class_stats['total_images']} images, {class_stats['total_annotations']} annotations")
    
    # Détecter les problèmes potentiels
    names_lower = {}
    duplicates = []
    for cat in categories:
        lower_name = cat["name"].lower().strip()
        if lower_name in names_lower:
            duplicates.append((names_lower[lower_name], cat["name"]))
        else:
            names_lower[lower_name] = cat["name"]
    
    if duplicates:
        print(f"\n   ⚠️  Doublons potentiels détectés:")
        for name1, name2 in duplicates:
            print(f"      • '{name1}' ↔ '{name2}'")


def get_user_input(prompt: str, default: str = None) -> str:
    """
    Récupère l'input utilisateur (compatible SSH)
    
    Args:
        prompt: Message à afficher
        default: Valeur par défaut si entrée vide
    
    Returns:
        Input de l'utilisateur
    """
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    
    try:
        response = input(full_prompt).strip()
        if not response and default:
            return default
        return response
    except EOFError:
        # En cas de pipe ou redirection
        if default:
            return default
        return ""


def get_yes_no(prompt: str, default: bool = False) -> bool:
    """
    Demande une confirmation oui/non
    
    Args:
        prompt: Question à poser
        default: Valeur par défaut (True=oui, False=non)
    
    Returns:
        True si oui, False si non
    """
    default_str = "O/n" if default else "o/N"
    
    while True:
        response = get_user_input(f"{prompt} ({default_str})", "").lower()
        
        if response == "":
            return default
        elif response in ["o", "oui", "y", "yes", "1"]:
            return True
        elif response in ["n", "non", "no", "0"]:
            return False
        else:
            print("   ❓ Répondez par 'o' (oui) ou 'n' (non)")


def interactive_merge_classes(categories: List[Dict]) -> List[List[int]]:
    """
    Demande à l'utilisateur quelles classes fusionner
    
    Args:
        categories: Liste des catégories [{"id": 0, "name": "pig", ...}, ...]
    
    Returns:
        Liste des groupes de fusion [[0, 1], [2, 3], ...]
        Chaque groupe contient les IDs à fusionner ensemble
    """
    if len(categories) <= 1:
        print("\n   ℹ️  Une seule classe, pas de fusion possible.")
        return []
    
    merge_groups = []
    remaining_ids = {cat["id"] for cat in categories}
    id_to_name = {cat["id"]: cat["name"] for cat in categories}
    
    print(f"\n{'─'*60}")
    print("🔀 FUSION DES CLASSES")
    print(f"{'─'*60}")
    
    if not get_yes_no("   Voulez-vous fusionner des classes ?", default=False):
        return []
    
    while True:
        if len(remaining_ids) <= 1:
            print("\n   ℹ️  Plus assez de classes à fusionner.")
            break
        
        # Afficher les classes restantes
        print(f"\n   Classes disponibles:")
        for cat_id in sorted(remaining_ids):
            print(f"      ID {cat_id}: {id_to_name[cat_id]}")
        
        print(f"\n   💡 Entrez les IDs ou noms des classes à fusionner, séparés par des virgules")
        print(f"      Exemples: '0,1' ou 'pig,Pig' ou '0,1,2' pour fusionner 3 classes")
        
        response = get_user_input("   Classes à fusionner (ou 'q' pour terminer)")
        
        if response.lower() in ['q', 'quit', 'fin', '']:
            break
        
        # Parser la réponse
        ids_to_merge = set()
        parts = [p.strip() for p in response.split(",")]
        
        for part in parts:
            # Essayer de parser comme ID numérique
            if part.isdigit():
                cat_id = int(part)
                if cat_id in remaining_ids:
                    ids_to_merge.add(cat_id)
                else:
                    print(f"   ⚠️  ID {cat_id} non trouvé ou déjà fusionné")
            else:
                # Chercher par nom (case insensitive)
                found = False
                for cat_id, name in id_to_name.items():
                    if name.lower() == part.lower() and cat_id in remaining_ids:
                        ids_to_merge.add(cat_id)
                        found = True
                        break
                if not found:
                    print(f"   ⚠️  Classe '{part}' non trouvée ou déjà fusionnée")
        
        if len(ids_to_merge) < 2:
            print("   ❌ Il faut au moins 2 classes pour fusionner")
            continue
        
        # Confirmer la fusion
        names_to_merge = [f"{id_to_name[i]} (ID:{i})" for i in sorted(ids_to_merge)]
        print(f"\n   📋 Fusion proposée: {' + '.join(names_to_merge)}")
        
        if get_yes_no("   Confirmer cette fusion ?", default=True):
            merge_groups.append(sorted(ids_to_merge))
            remaining_ids -= ids_to_merge
            print(f"   ✅ Fusion enregistrée!")
            
            # Demander si autre fusion
            if len(remaining_ids) >= 2:
                if not get_yes_no("\n   Effectuer une autre fusion ?", default=False):
                    break
            else:
                break
        else:
            print("   ↩️  Fusion annulée")
    
    # Résumé des fusions
    if merge_groups:
        print(f"\n   {'─'*50}")
        print(f"   📋 Résumé des fusions ({len(merge_groups)}):")
        for i, group in enumerate(merge_groups, 1):
            names = [f"{id_to_name[id]}" for id in group]
            print(f"      {i}. {' + '.join(names)} → fusionnées")
    
    return merge_groups


def interactive_rename_classes(categories: List[Dict], merge_groups: List[List[int]]) -> Dict[int, str]:
    """
    Demande à l'utilisateur de renommer les classes
    
    Affiche clairement le nom actuel (ou les noms fusionnés) pour chaque classe
    
    Args:
        categories: Liste des catégories originales
        merge_groups: Groupes de fusion [[0, 1], ...]
    
    Returns:
        Dictionnaire {new_id: new_name}
    """
    id_to_name = {cat["id"]: cat["name"] for cat in categories}
    id_to_annotations = {cat["id"]: cat["annotations"] for cat in categories}
    
    # Construire la liste des classes finales après fusion
    # Chaque entrée: {"ids": [0, 1], "names": ["pig", "Pig"], "annotations": 6315}
    final_classes = []
    
    # IDs qui font partie d'une fusion
    merged_ids = set()
    for group in merge_groups:
        merged_ids.update(group)
        names = [id_to_name[id] for id in group]
        total_annotations = sum(id_to_annotations.get(id, 0) for id in group)
        final_classes.append({
            "ids": group,
            "names": names,
            "annotations": total_annotations,
            "merged": True
        })
    
    # IDs non fusionnés
    for cat in categories:
        if cat["id"] not in merged_ids:
            final_classes.append({
                "ids": [cat["id"]],
                "names": [cat["name"]],
                "annotations": cat["annotations"],
                "merged": False
            })
    
    # Trier par premier ID
    final_classes.sort(key=lambda x: x["ids"][0])
    
    print(f"\n{'─'*60}")
    print("✏️  RENOMMAGE DES CLASSES")
    print(f"{'─'*60}")
    
    if not get_yes_no("   Voulez-vous renommer des classes ?", default=False):
        # Retourner les noms par défaut (premier nom de chaque groupe)
        renames = {}
        for new_id, cls in enumerate(final_classes):
            renames[new_id] = cls["names"][0]
        return renames
    
    renames = {}
    
    print(f"\n   💡 Pour chaque classe, entrez le nouveau nom ou appuyez sur Entrée pour garder l'actuel")
    print(f"   {'─'*50}")
    
    for new_id, cls in enumerate(final_classes):
        if cls["merged"]:
            # Classe fusionnée - afficher tous les noms d'origine
            names_str = " + ".join(cls["names"])
            current_display = f"[FUSIONNÉE: {names_str}]"
            default_name = cls["names"][0]  # Premier nom par défaut
        else:
            # Classe non fusionnée
            current_display = cls["names"][0]
            default_name = cls["names"][0]
        
        print(f"\n   Classe {new_id}:")
        print(f"      Nom actuel: {current_display}")
        print(f"      Annotations: {cls['annotations']}")
        
        new_name = get_user_input(f"      Nouveau nom", default=default_name)
        
        if new_name != default_name:
            print(f"      ✅ Renommée: {current_display} → '{new_name}'")
        else:
            print(f"      ✓ Conservé: '{new_name}'")
        
        renames[new_id] = new_name
    
    # Résumé
    print(f"\n   {'─'*50}")
    print(f"   📋 Configuration finale des classes:")
    for new_id, new_name in renames.items():
        print(f"      ID {new_id}: {new_name}")
    
    return renames


def apply_class_changes(dataset_path: Path, merge_groups: List[List[int]], renames: Dict[int, str]) -> Dict:
    """
    Applique les fusions et renommages au dataset COCO
    
    Args:
        dataset_path: Chemin du dataset
        merge_groups: Groupes de fusion [[0, 1], ...]
        renames: Mapping {new_id: new_name}
    
    Returns:
        Dictionnaire avec les infos des classes finales
    """
    print(f"\n{'='*70}")
    print(f"⚙️  APPLICATION DES MODIFICATIONS")
    print(f"{'='*70}")
    
    # Construire le mapping old_id -> new_id
    old_to_new = {}
    
    # D'abord, récupérer tous les IDs originaux
    original_ids = set()
    for split in ["train", "valid", "test"]:
        ann_file = dataset_path / split / "_annotations.coco.json"
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                coco = json.load(f)
            for cat in coco.get("categories", []):
                original_ids.add(cat["id"])
            break
    
    # IDs fusionnés
    merged_ids = set()
    new_id_counter = 0
    
    for group in merge_groups:
        merged_ids.update(group)
        for old_id in group:
            old_to_new[old_id] = new_id_counter
        new_id_counter += 1
    
    # IDs non fusionnés
    for old_id in sorted(original_ids):
        if old_id not in merged_ids:
            old_to_new[old_id] = new_id_counter
            new_id_counter += 1
    
    print(f"   📋 Mapping des IDs:")
    for old_id, new_id in sorted(old_to_new.items()):
        new_name = renames.get(new_id, f"class_{new_id}")
        print(f"      {old_id} → {new_id} ({new_name})")
    
    # Appliquer à chaque split
    for split in ["train", "valid", "test"]:
        ann_file = dataset_path / split / "_annotations.coco.json"
        if not ann_file.exists():
            continue
        
        print(f"\n   📁 Traitement de {split}...")
        
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        
        # Créer le backup
        backup_file = ann_file.with_suffix('.json.backup_classes')
        if not backup_file.exists():
            shutil.copy(ann_file, backup_file)
            print(f"      💾 Backup créé: {backup_file.name}")
        
        # Nouvelles catégories
        new_categories = []
        for new_id, new_name in sorted(renames.items()):
            new_categories.append({
                "id": new_id,
                "name": new_name,
                "supercategory": ""
            })
        
        # Mettre à jour les annotations
        updated_count = 0
        for ann in coco.get("annotations", []):
            old_cat_id = ann["category_id"]
            if old_cat_id in old_to_new:
                ann["category_id"] = old_to_new[old_cat_id]
                updated_count += 1
        
        # Sauvegarder
        coco["categories"] = new_categories
        
        with open(ann_file, 'w') as f:
            json.dump(coco, f, indent=2)
        
        print(f"      ✅ {updated_count} annotations mises à jour")
        print(f"      ✅ {len(new_categories)} catégories définies")
    
    # Retourner les infos des classes finales
    class_info = {
        "classes": {},
        "old_to_new_mapping": old_to_new
    }
    
    for new_id, new_name in renames.items():
        class_info["classes"][new_id] = {
            "name": new_name,
            "supercategory": ""
        }
    
    print(f"\n✅ Modifications appliquées avec succès!")
    
    return class_info


def interactive_class_management(dataset_path: Path, dataset_name: str) -> Dict:
    """
    Gestion interactive complète des classes d'un dataset
    
    1. Affiche les stats des classes
    2. Propose la fusion
    3. Propose le renommage
    4. Applique les modifications
    
    Args:
        dataset_path: Chemin du dataset
        dataset_name: Nom du dataset
    
    Returns:
        Dictionnaire avec les infos des classes finales
    """
    if not Config.INTERACTIVE_MODE:
        # Mode non-interactif: retourner les classes telles quelles
        categories = extract_classes_from_dataset(dataset_path)
        return {
            "classes": {cat["id"]: {"name": cat["name"]} for cat in categories}
        }
    
    # Analyser les classes
    class_stats = analyze_dataset_classes(dataset_path)
    
    # Afficher le tableau
    display_classes_table(dataset_name, class_stats)
    
    categories = class_stats["categories"]
    
    if not categories:
        print("   ❌ Aucune classe trouvée dans le dataset!")
        return {"classes": {}}
    
    # Si une seule classe et pas de problème, passer directement
    if len(categories) == 1:
        print(f"\n   ✅ Une seule classe détectée: '{categories[0]['name']}'")
        if get_yes_no("   Voulez-vous la renommer ?", default=False):
            new_name = get_user_input(f"   Nouveau nom", default=categories[0]['name'])
            if new_name != categories[0]['name']:
                renames = {0: new_name}
                return apply_class_changes(dataset_path, [], renames)
        
        return {
            "classes": {0: {"name": categories[0]["name"]}}
        }
    
    # Fusion interactive
    merge_groups = interactive_merge_classes(categories)
    
    # Renommage interactif
    renames = interactive_rename_classes(categories, merge_groups)
    
    # Appliquer les modifications si nécessaire
    if merge_groups or any(renames.get(i) != categories[i]["name"] for i in range(len(categories)) if i < len(renames)):
        # Vérifier s'il y a vraiment des changements
        has_changes = bool(merge_groups)
        
        if not has_changes:
            # Vérifier les renommages
            id_to_name = {cat["id"]: cat["name"] for cat in categories}
            for new_id, new_name in renames.items():
                if new_id in id_to_name and id_to_name[new_id] != new_name:
                    has_changes = True
                    break
        
        if has_changes:
            print(f"\n{'─'*60}")
            if get_yes_no("   📝 Appliquer ces modifications au dataset ?", default=True):
                return apply_class_changes(dataset_path, merge_groups, renames)
            else:
                print("   ↩️  Modifications annulées, classes conservées telles quelles")
    
    # Pas de changements ou annulé - retourner les classes originales
    return {
        "classes": {cat["id"]: {"name": cat["name"]} for cat in categories}
    }


# ============================================================================
# 5. VALIDATION ET CORRECTION DES DATASETS COCO
# ============================================================================

def validate_coco_dataset(dataset_path: Path) -> bool:
    """
    Valide que le dataset COCO a des category_id valides (continus à partir de 0)
    
    Cette validation est CRUCIALE pour éviter l'erreur CUDA:
    "device-side assert triggered"
    
    Returns:
        True si valide, False sinon
    """
    print(f"\n{'='*70}")
    print(f"🔍 VALIDATION DU DATASET COCO")
    print(f"{'='*70}")
    print(f"   Chemin: {dataset_path}")
    
    issues = []
    all_valid = True
    
    for split in ["train", "valid", "test"]:
        ann_file = dataset_path / split / "_annotations.coco.json"
        if not ann_file.exists():
            continue
        
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        
        categories = coco.get("categories", [])
        annotations = coco.get("annotations", [])
        images = coco.get("images", [])
        
        # Vérifier les IDs de catégories
        cat_ids = sorted([c["id"] for c in categories])
        expected_ids = list(range(len(categories)))
        
        print(f"\n   📁 {split.upper()}:")
        print(f"      Images:       {len(images)}")
        print(f"      Annotations:  {len(annotations)}")
        print(f"      Catégories:   {len(categories)}")
        print(f"      IDs trouvés:  {cat_ids}")
        print(f"      IDs attendus: {expected_ids}")
        
        # Vérifier si les IDs commencent à 0
        if cat_ids and cat_ids[0] != 0:
            issues.append(f"❌ {split}: Les category_id commencent à {cat_ids[0]} au lieu de 0")
            print(f"      ⚠️  PROBLÈME: Les IDs ne commencent pas à 0!")
            all_valid = False
        
        # Vérifier si les IDs sont continus
        if cat_ids != expected_ids:
            issues.append(f"❌ {split}: Les category_id ne sont pas continus: {cat_ids}")
            print(f"      ⚠️  PROBLÈME: Les IDs ne sont pas continus!")
            all_valid = False
        
        # Vérifier les annotations
        annotation_cat_ids = set(ann["category_id"] for ann in annotations)
        category_ids_set = set(cat_ids)
        invalid_cats = annotation_cat_ids - category_ids_set
        
        if invalid_cats:
            issues.append(f"❌ {split}: Annotations avec category_id invalides: {invalid_cats}")
            print(f"      ⚠️  PROBLÈME: Annotations avec IDs invalides: {invalid_cats}")
            all_valid = False
        
        # Afficher les catégories
        if categories:
            print(f"      Classes:")
            for cat in categories:
                print(f"         ID {cat['id']}: {cat['name']}")
        
        if all_valid:
            print(f"      ✅ Valide!")
    
    if issues:
        print(f"\n{'='*70}")
        print(f"❌ PROBLÈMES DÉTECTÉS ({len(issues)}):")
        print(f"{'='*70}")
        for issue in issues:
            print(f"   {issue}")
        print(f"\n💡 Ces problèmes causent l'erreur CUDA 'device-side assert triggered'")
        print(f"   Le script va corriger automatiquement les IDs...")
        return False
    
    print(f"\n✅ Dataset COCO valide! Les category_id sont corrects (0, 1, 2, ...)")
    return True


def fix_coco_category_ids(dataset_path: Path) -> Dict:
    """
    Corrige les category_id pour qu'ils soient continus à partir de 0
    
    Cette correction est ESSENTIELLE pour éviter l'erreur CUDA.
    
    Returns:
        Dictionnaire avec le mapping old_id -> new_id
    """
    print(f"\n{'='*70}")
    print(f"🔧 CORRECTION DES CATEGORY_ID")
    print(f"{'='*70}")
    
    global_mapping = {}
    
    for split in ["train", "valid", "test"]:
        ann_file = dataset_path / split / "_annotations.coco.json"
        if not ann_file.exists():
            continue
        
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        
        categories = coco.get("categories", [])
        annotations = coco.get("annotations", [])
        
        if not categories:
            continue
        
        # Créer le mapping old_id -> new_id (continu à partir de 0)
        old_ids = sorted([c["id"] for c in categories])
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
        
        print(f"\n   📁 {split.upper()}:")
        print(f"      Mapping: {old_ids} → {list(range(len(old_ids)))}")
        
        # Sauvegarder le mapping global
        for old_id, new_id in id_mapping.items():
            cat_name = next((c["name"] for c in categories if c["id"] == old_id), "unknown")
            global_mapping[old_id] = {"new_id": new_id, "name": cat_name}
            print(f"         {old_id} → {new_id} ({cat_name})")
        
        # Mettre à jour les catégories
        for cat in categories:
            cat["id"] = id_mapping[cat["id"]]
        
        # Mettre à jour les annotations
        updated_count = 0
        for ann in annotations:
            old_cat_id = ann["category_id"]
            if old_cat_id in id_mapping:
                ann["category_id"] = id_mapping[old_cat_id]
                updated_count += 1
        
        print(f"      Annotations mises à jour: {updated_count}")
        
        # Créer un backup avant de modifier
        backup_file = ann_file.with_suffix('.json.backup')
        if not backup_file.exists():
            shutil.copy(ann_file, backup_file)
            print(f"      💾 Backup créé: {backup_file.name}")
        
        # Sauvegarder le fichier corrigé
        with open(ann_file, 'w') as f:
            json.dump(coco, f, indent=2)
        
        print(f"      ✅ Fichier corrigé sauvegardé!")
    
    print(f"\n✅ Correction terminée!")
    print(f"   Les category_id sont maintenant continus à partir de 0")
    
    return global_mapping


def ensure_valid_coco_dataset(dataset_path: Path) -> bool:
    """
    S'assure que le dataset COCO est valide, corrige si nécessaire.
    
    Returns:
        True si le dataset est valide (ou a été corrigé avec succès)
    """
    # Première validation
    if validate_coco_dataset(dataset_path):
        return True
    
    # Correction automatique
    print("\n⚠️ Dataset invalide détecté, correction automatique en cours...")
    fix_coco_category_ids(dataset_path)
    
    # Re-validation après correction
    print("\n🔄 Re-validation après correction...")
    if validate_coco_dataset(dataset_path):
        print("✅ Dataset corrigé avec succès!")
        return True
    else:
        print("❌ Échec de la correction automatique!")
        return False


# ============================================================================
# 6. FUSION DES DATASETS COCO (MODE MERGED)
# ============================================================================

def analyze_all_datasets_classes(dataset_paths: List[Path], dataset_names: List[str]) -> Dict:
    """
    Analyse les classes de tous les datasets pour le mode merged
    
    Note: Les classes vides ont déjà été supprimées lors du téléchargement,
          mais on filtre quand même par sécurité.
    """
    result = {
        "all_categories": [],
        "by_dataset": {},
        "unique_names": set(),
        "removed_empty": []  # Pour tracking
    }
    
    for dataset_path, dataset_name in zip(dataset_paths, dataset_names):
        stats = analyze_dataset_classes(dataset_path)
        result["by_dataset"][dataset_name] = stats["categories"]
        
        for cat in stats["categories"]:
            # Ignorer les classes sans annotations (sécurité supplémentaire)
            if cat["annotations"] == 0:
                result["removed_empty"].append({
                    "name": cat["name"],
                    "source": dataset_name
                })
                continue
            
            result["all_categories"].append({
                "name": cat["name"],
                "source": dataset_name,
                "images": cat["images"],
                "annotations": cat["annotations"],
                "original_id": cat["id"]
            })
            result["unique_names"].add(cat["name"])
    
    return result

def display_merged_classes_table(all_stats: Dict):
    """
    Affiche un tableau consolidé de toutes les classes pour le mode merged
    """
    all_cats = all_stats["all_categories"]
    
    print(f"\n{'='*70}")
    print(f"📦 CLASSES DE TOUS LES DATASETS (MODE MERGED)")
    print(f"{'='*70}")
    
    if not all_cats:
        print("   ❌ Aucune classe trouvée!")
        return
    
    # Calculer les largeurs
    max_name_len = max(len(cat["name"]) for cat in all_cats)
    max_name_len = max(max_name_len, 10)
    max_source_len = max(len(cat["source"]) for cat in all_cats)
    max_source_len = max(max_source_len, 8)
    
    # En-tête
    print(f"\n   ┌{'─'*(max_name_len+2)}┬{'─'*(max_source_len+2)}┬{'─'*10}┬{'─'*14}┐")
    print(f"   │ {'Nom':^{max_name_len}} │ {'Source':^{max_source_len}} │ {'Images':^8} │ {'Annotations':^12} │")
    print(f"   ├{'─'*(max_name_len+2)}┼{'─'*(max_source_len+2)}┼{'─'*10}┼{'─'*14}┤")
    
    # Trier par nom pour regrouper les similaires
    sorted_cats = sorted(all_cats, key=lambda x: (x["name"].lower(), x["source"]))
    
    for cat in sorted_cats:
        print(f"   │ {cat['name']:<{max_name_len}} │ {cat['source']:<{max_source_len}} │ {cat['images']:>8} │ {cat['annotations']:>12} │")
    
    print(f"   └{'─'*(max_name_len+2)}┴{'─'*(max_source_len+2)}┴{'─'*10}┴{'─'*14}┘")
    
    # Détecter les doublons potentiels (casse différente)
    names_lower = {}
    duplicates = []
    for cat in all_cats:
        lower_name = cat["name"].lower().strip()
        key = lower_name
        if key in names_lower:
            if names_lower[key] != cat["name"]:
                duplicates.append((names_lower[key], cat["name"]))
        else:
            names_lower[key] = cat["name"]
    
    # Détecter les classes avec même nom dans différents datasets
    name_sources = {}
    for cat in all_cats:
        name = cat["name"]
        if name not in name_sources:
            name_sources[name] = []
        name_sources[name].append(cat["source"])
    
    shared_classes = {name: sources for name, sources in name_sources.items() if len(sources) > 1}
    
    if duplicates:
        print(f"\n   ⚠️  Doublons potentiels (casse différente):")
        seen = set()
        for name1, name2 in duplicates:
            pair = tuple(sorted([name1, name2]))
            if pair not in seen:
                print(f"      • '{name1}' ↔ '{name2}'")
                seen.add(pair)
    
    if shared_classes:
        print(f"\n   ℹ️  Classes partagées entre datasets:")
        for name, sources in shared_classes.items():
            print(f"      • '{name}' dans: {', '.join(sources)}")
    
    # Afficher les classes vides qui ont été ignorées
    if all_stats.get("removed_empty"):
        removed = all_stats["removed_empty"]
        print(f"\n   🧹 {len(removed)} classe(s) vide(s) ignorée(s) automatiquement:")
        # Regrouper par nom pour éviter les doublons
        removed_names = {}
        for item in removed:
            name = item['name']
            if name not in removed_names:
                removed_names[name] = []
            removed_names[name].append(item['source'])
        
        for name, sources in sorted(removed_names.items()):
            print(f"      • '{name}' (depuis: {', '.join(sources)})")
    
    print(f"\n   📊 Total: {len(all_stats['unique_names'])} nom(s) unique(s), {len(all_cats)} entrée(s)")

def interactive_merge_classes_merged_mode(all_stats: Dict) -> Tuple[List[List[Tuple[str, str]]], Dict[Tuple[str, str], str]]:
    """
    Gestion interactive des classes en mode merged (plusieurs datasets)
    
    Chaque classe est identifiée par son couple (nom, source) pour permettre
    de renommer différemment des classes de même nom venant de datasets différents.
    
    Returns:
        (merge_groups, renames)
        merge_groups: Liste des groupes à fusionner [[("pig", "ds1"), ("Pig", "ds2")], ...]
        renames: Mapping {(old_name, source): new_name}
    """
    all_cats = all_stats["all_categories"]
    
    if not all_cats:
        return [], {}
    
    # Créer une liste de toutes les classes avec leur source (chaque entrée est unique)
    all_class_entries = []
    for cat in all_cats:
        all_class_entries.append({
            "name": cat["name"],
            "source": cat["source"],
            "annotations": cat["annotations"],
            "images": cat["images"],
            "key": (cat["name"], cat["source"])  # Clé unique
        })
    
    # === FUSION ===
    print(f"\n{'─'*60}")
    print("🔀 FUSION DES CLASSES (MODE MERGED)")
    print(f"{'─'*60}")
    
    merge_groups = []  # Liste de groupes, chaque groupe = liste de (name, source)
    remaining_entries = {entry["key"] for entry in all_class_entries}
    
    if len(all_class_entries) > 1 and get_yes_no("   Voulez-vous fusionner des classes ?", default=False):
        while len(remaining_entries) >= 2:
            print(f"\n   Classes disponibles:")
            # Trier par nom puis par source
            sorted_remaining = sorted(remaining_entries, key=lambda x: (x[0].lower(), x[1]))
            for name, source in sorted_remaining:
                # Trouver les annotations
                entry = next(e for e in all_class_entries if e["key"] == (name, source))
                print(f"      • '{name}' [{source}] ({entry['annotations']} ann.)")
            
            print(f"\n   💡 Entrez les classes à fusionner au format: nom:source")
            print(f"      Exemple: 'Rat:dataset1,Rat:dataset2' ou 'pig:ds1,Pig:ds2'")
            print(f"      Ou juste le nom si unique: 'pig,Pig'")
            
            response = get_user_input("   Classes à fusionner (ou 'q' pour terminer)")
            
            if response.lower() in ['q', 'quit', 'fin', '']:
                break
            
            # Parser la réponse
            entries_to_merge = set()
            parts = [p.strip() for p in response.split(",")]
            
            for part in parts:
                if ':' in part:
                    # Format explicite: nom:source
                    name_part, source_part = part.rsplit(':', 1)
                    name_part = name_part.strip()
                    source_part = source_part.strip()
                    
                    # Chercher la correspondance
                    found = False
                    for key in remaining_entries:
                        if key[0].lower() == name_part.lower() and key[1].lower() == source_part.lower():
                            entries_to_merge.add(key)
                            found = True
                            break
                    if not found:
                        print(f"   ⚠️  Classe '{name_part}' du dataset '{source_part}' non trouvée")
                else:
                    # Format simple: juste le nom
                    # Chercher toutes les classes avec ce nom
                    matching = [key for key in remaining_entries if key[0].lower() == part.lower()]
                    if len(matching) == 0:
                        print(f"   ⚠️  Classe '{part}' non trouvée")
                    elif len(matching) == 1:
                        entries_to_merge.add(matching[0])
                    else:
                        # Plusieurs classes avec le même nom - demander précision
                        print(f"   ⚠️  Plusieurs classes '{part}' trouvées:")
                        for name, source in matching:
                            print(f"         • '{name}' [{source}]")
                        print(f"      Précisez avec le format 'nom:source'")
            
            if len(entries_to_merge) < 2:
                print("   ❌ Il faut au moins 2 classes pour fusionner")
                continue
            
            # Confirmer
            merge_display = [f"'{name}' [{source}]" for name, source in sorted(entries_to_merge)]
            print(f"\n   📋 Fusion proposée: {' + '.join(merge_display)}")
            
            if get_yes_no("   Confirmer ?", default=True):
                merge_groups.append(list(entries_to_merge))
                remaining_entries -= entries_to_merge
                print(f"   ✅ Fusion enregistrée!")
                
                if len(remaining_entries) >= 2:
                    if not get_yes_no("\n   Autre fusion ?", default=False):
                        break
            else:
                print("   ↩️  Annulé")
    
    # === RENOMMAGE ===
    print(f"\n{'─'*60}")
    print("✏️  RENOMMAGE DES CLASSES (MODE MERGED)")
    print(f"{'─'*60}")
    
    # Construire la liste finale des classes
    final_classes = []
    
    # Classes fusionnées
    merged_keys = set()
    for group in merge_groups:
        total_ann = sum(
            next(e["annotations"] for e in all_class_entries if e["key"] == key)
            for key in group
        )
        final_classes.append({
            "keys": group,  # Liste de (name, source)
            "annotations": total_ann,
            "merged": True
        })
        merged_keys.update(group)
    
    # Classes non fusionnées (chacune séparément!)
    for entry in all_class_entries:
        if entry["key"] not in merged_keys:
            final_classes.append({
                "keys": [entry["key"]],
                "annotations": entry["annotations"],
                "merged": False
            })
    
    # Trier par premier nom
    final_classes.sort(key=lambda x: (x["keys"][0][0].lower(), x["keys"][0][1]))
    
    renames = {}  # {(name, source): new_name}
    
    if get_yes_no("   Voulez-vous renommer des classes ?", default=False):
        print(f"\n   💡 Pour chaque classe, entrez le nouveau nom ou Entrée pour conserver")
        
        for cls in final_classes:
            if cls["merged"]:
                # Classe fusionnée - afficher tous les noms/sources d'origine
                names_str = " + ".join([f"'{name}' [{source}]" for name, source in cls["keys"]])
                current_display = f"[FUSIONNÉE: {names_str}]"
                default_name = cls["keys"][0][0]  # Premier nom par défaut
            else:
                # Classe non fusionnée - afficher nom et source
                name, source = cls["keys"][0]
                current_display = f"'{name}' [{source}]"
                default_name = name
            
            print(f"\n   Classe:")
            print(f"      Actuel: {current_display}")
            print(f"      Annotations: {cls['annotations']}")
            
            new_name = get_user_input(f"      Nouveau nom", default=default_name)
            
            # Stocker le renommage pour toutes les clés du groupe
            for key in cls["keys"]:
                renames[key] = new_name
            
            if new_name != default_name:
                print(f"      ✅ → '{new_name}'")
    else:
        # Pas de renommage interactif
        # ATTENTION: ici on doit décider quoi faire pour les classes de même nom
        # Par défaut, on garde le nom original (ce qui fusionnera automatiquement les mêmes noms)
        for cls in final_classes:
            default_name = cls["keys"][0][0]
            for key in cls["keys"]:
                renames[key] = default_name
    
    # Résumé
    print(f"\n   {'─'*50}")
    print(f"   📋 Configuration finale:")
    
    # Regrouper par nom final pour afficher
    final_names_to_sources = {}
    for key, new_name in renames.items():
        if new_name not in final_names_to_sources:
            final_names_to_sources[new_name] = []
        final_names_to_sources[new_name].append(key)
    
    for i, (final_name, sources) in enumerate(sorted(final_names_to_sources.items())):
        if len(sources) > 1:
            sources_str = ", ".join([f"'{n}' [{s}]" for n, s in sources])
            print(f"      ID {i}: '{final_name}' (fusion de: {sources_str})")
        else:
            name, source = sources[0]
            if name != final_name:
                print(f"      ID {i}: '{final_name}' (renommé depuis '{name}' [{source}])")
            else:
                print(f"      ID {i}: '{final_name}' [{source}]")
    
    return merge_groups, renames

def merge_coco_datasets(dataset_paths: List[Path], output_path: Path, 
                        merge_groups: List[List[Tuple[str, str]]] = None, 
                        renames: Dict[Tuple[str, str], str] = None) -> Dict:
    """
    Fusionne plusieurs datasets COCO en un seul
    
    Args:
        dataset_paths: Liste des chemins des datasets
        output_path: Chemin de sortie
        merge_groups: Groupes de (name, source) à fusionner (optionnel)
        renames: Mapping {(old_name, source): new_name} (optionnel)
    
    Returns:
        Dictionnaire avec le mapping des classes
    """
    print(f"\n{'='*70}")
    print(f"🔀 FUSION DES DATASETS")
    print(f"{'='*70}")
    
    # D'abord, valider/corriger chaque dataset source
    print("\n📋 Validation des datasets sources...")
    for dataset_path in dataset_paths:
        ensure_valid_coco_dataset(dataset_path)
    
    # Construire le mapping (name, source) -> ID final
    key_to_final_id = {}  # {(name, source): final_id}
    final_id_to_name = {}  # {final_id: final_name}
    final_id_counter = 0
    
    if renames:
        # D'abord, regrouper par nom final
        final_name_to_keys = {}
        for key, final_name in renames.items():
            if final_name not in final_name_to_keys:
                final_name_to_keys[final_name] = []
            final_name_to_keys[final_name].append(key)
        
        # Assigner un ID à chaque nom final unique
        for final_name in sorted(final_name_to_keys.keys()):
            keys = final_name_to_keys[final_name]
            for key in keys:
                key_to_final_id[key] = final_id_counter
            final_id_to_name[final_id_counter] = final_name
            final_id_counter += 1
    
    merged = {
        "train": {"images": [], "annotations": [], "categories": []},
        "valid": {"images": [], "annotations": [], "categories": []},
        "test": {"images": [], "annotations": [], "categories": []}
    }
    
    class_mapping = {
        "classes": {},
        "source_datasets": {}
    }
    
    image_id_offset = 0
    annotation_id_offset = 0
    
    for dataset_idx, dataset_path in enumerate(dataset_paths):
        dataset_name = dataset_path.name
        print(f"\n📁 Traitement: {dataset_name}")
        
        class_mapping["source_datasets"][dataset_name] = {
            "path": str(dataset_path),
            "original_categories": {}
        }
        
        for split in ["train", "valid", "test"]:
            split_dir = dataset_path / split
            ann_file = split_dir / "_annotations.coco.json"
            
            if not ann_file.exists():
                continue
            
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            local_id_to_final_id = {}
            
            for cat in coco_data.get("categories", []):
                cat_name = cat["name"]
                original_id = cat["id"]
                key = (cat_name, dataset_name)
                
                # Déterminer le nom final et l'ID final
                if renames and key in key_to_final_id:
                    final_id = key_to_final_id[key]
                    final_name = final_id_to_name[final_id]
                elif renames:
                    # Clé non trouvée dans renames - chercher par nom seul (compatibilité)
                    # Chercher si une clé avec ce nom existe
                    found = False
                    for rkey, rname in renames.items():
                        if rkey[0] == cat_name:
                            final_name = rname
                            if final_name in [final_id_to_name.get(fid) for fid in final_id_to_name]:
                                final_id = next(fid for fid, fname in final_id_to_name.items() if fname == final_name)
                            else:
                                final_id = final_id_counter
                                final_id_to_name[final_id] = final_name
                                final_id_counter += 1
                            key_to_final_id[key] = final_id
                            found = True
                            break
                    
                    if not found:
                        # Nouveau nom non renommé
                        final_name = cat_name
                        # Vérifier si ce nom final existe déjà
                        existing_id = next((fid for fid, fname in final_id_to_name.items() if fname == final_name), None)
                        if existing_id is not None:
                            final_id = existing_id
                        else:
                            final_id = final_id_counter
                            final_id_to_name[final_id] = final_name
                            final_id_counter += 1
                        key_to_final_id[key] = final_id
                else:
                    # Pas de renames - utiliser le nom tel quel
                    final_name = cat_name
                    # Vérifier si ce nom existe déjà
                    existing_id = next((fid for fid, fname in final_id_to_name.items() if fname == final_name), None)
                    if existing_id is not None:
                        final_id = existing_id
                    else:
                        final_id = final_id_counter
                        final_id_to_name[final_id] = final_name
                        final_id_counter += 1
                    key_to_final_id[key] = final_id
                
                local_id_to_final_id[original_id] = final_id
                
                print(f"   📎 '{cat_name}' [{dataset_name}] (ID:{original_id}) → '{final_name}' (ID:{final_id})")
                
                # Ajouter à class_mapping si nouveau
                if final_id not in class_mapping["classes"]:
                    class_mapping["classes"][final_id] = {
                        "name": final_name,
                        "supercategory": cat.get("supercategory", ""),
                        "source_datasets": [dataset_name],
                        "original_names": [(cat_name, dataset_name)]
                    }
                else:
                    if dataset_name not in class_mapping["classes"][final_id]["source_datasets"]:
                        class_mapping["classes"][final_id]["source_datasets"].append(dataset_name)
                    class_mapping["classes"][final_id]["original_names"].append((cat_name, dataset_name))
                
                class_mapping["source_datasets"][dataset_name]["original_categories"][original_id] = {
                    "name": cat_name,
                    "final_name": final_name,
                    "final_id": final_id
                }
            
            # Images
            local_image_map = {}
            for img in coco_data.get("images", []):
                old_id = img["id"]
                new_id = image_id_offset + old_id
                local_image_map[old_id] = new_id
                
                new_img = img.copy()
                new_img["id"] = new_id
                new_img["original_file"] = img["file_name"]
                new_img["source_dataset"] = dataset_name
                
                merged[split]["images"].append(new_img)
            
            # Annotations
            for ann in coco_data.get("annotations", []):
                new_ann = ann.copy()
                new_ann["id"] = annotation_id_offset + ann["id"]
                new_ann["image_id"] = local_image_map[ann["image_id"]]
                new_ann["category_id"] = local_id_to_final_id[ann["category_id"]]
                
                merged[split]["annotations"].append(new_ann)
                annotation_id_offset += 1
            
            if coco_data.get("images"):
                image_id_offset = max(img["id"] for img in merged[split]["images"]) + 1
            
            print(f"   {split}: {len(coco_data.get('images', []))} images, {len(coco_data.get('annotations', []))} annotations")
    
    # Créer les catégories finales
    final_categories = []
    for final_id in sorted(final_id_to_name.keys()):
        final_name = final_id_to_name[final_id]
        supercategory = class_mapping["classes"].get(final_id, {}).get("supercategory", "")
        final_categories.append({
            "id": final_id,
            "name": final_name,
            "supercategory": supercategory
        })
    
    for split in ["train", "valid", "test"]:
        merged[split]["categories"] = final_categories.copy()
    
    # Créer le dataset fusionné
    print(f"\n💾 Création du dataset fusionné: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "valid", "test"]:
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        if not merged[split]["images"]:
            continue
        
        print(f"   📷 Copie des images {split}...")
        for img in merged[split]["images"]:
            source_dataset = img["source_dataset"]
            source_idx = [i for i, p in enumerate(dataset_paths) if p.name == source_dataset][0]
            source_file = dataset_paths[source_idx] / split / img["original_file"]
            
            if source_file.exists():
                dest_file = split_dir / img["original_file"]
                if dest_file.exists():
                    base, ext = os.path.splitext(img["original_file"])
                    new_name = f"{source_dataset}_{base}{ext}"
                    dest_file = split_dir / new_name
                    img["file_name"] = new_name
                else:
                    img["file_name"] = img["original_file"]
                
                shutil.copy2(source_file, dest_file)
            
            del img["original_file"]
            del img["source_dataset"]
        
        coco_output = {
            "images": merged[split]["images"],
            "annotations": merged[split]["annotations"],
            "categories": merged[split]["categories"]
        }
        
        ann_file = split_dir / "_annotations.coco.json"
        with open(ann_file, 'w') as f:
            json.dump(coco_output, f, indent=2)
        
        print(f"   ✅ {split}: {len(merged[split]['images'])} images, {len(merged[split]['annotations'])} annotations")
    
    print(f"\n📋 Catégories fusionnées ({len(final_categories)}):")
    for cat in final_categories:
        info = class_mapping["classes"].get(cat["id"], {})
        sources = info.get("source_datasets", ["?"])
        original_names = info.get("original_names", [])
        
        # Afficher les noms originaux si différents
        if original_names:
            unique_originals = list(set(name for name, src in original_names))
            if len(unique_originals) > 1 or (len(unique_originals) == 1 and unique_originals[0] != cat["name"]):
                originals_str = ", ".join([f"'{n}' [{s}]" for n, s in original_names])
                print(f"   ID {cat['id']}: '{cat['name']}' ← [{originals_str}]")
            else:
                print(f"   ID {cat['id']}: '{cat['name']}' (depuis: {', '.join(sources)})")
        else:
            print(f"   ID {cat['id']}: '{cat['name']}' (depuis: {', '.join(sources)})")
    
    # Valider le dataset fusionné
    print("\n🔍 Validation du dataset fusionné...")
    ensure_valid_coco_dataset(output_path)
    
    return class_mapping

def extract_classes_from_dataset(dataset_path: Path) -> list:
    """Extrait les classes depuis le fichier _annotations.coco.json"""
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
# 7. SAUVEGARDE DU MAPPING DES CLASSES
# ============================================================================

def save_class_mapping(output_dir: Path, class_info: dict, datasets_used: list = None):
    """Sauvegarde le mapping class_id → dataset/classe dans un fichier JSON"""
    mapping_file = output_dir / "class_mapping.json"
    
    mapping = {
        "created_at": datetime.now().isoformat(),
        "datasets_used": datasets_used or [],
        "total_classes": len(class_info.get("classes", class_info)),
        "classes": class_info.get("classes", class_info),
        "training_config": {
            "resolution": Config.RESOLUTION,
            "epochs": Config.EPOCHS,
            "model_size": Config.MODEL_SIZE,
            "early_stopping": Config.EARLY_STOPPING,
            "early_stopping_patience": Config.EARLY_STOPPING_PATIENCE,
            "batch_size": Config.BATCH_SIZE,
            "grad_accum_steps": Config.GRAD_ACCUM_STEPS,
            "learning_rate": Config.LR,
        }
    }
    
    if "source_datasets" in class_info:
        mapping["source_datasets"] = class_info["source_datasets"]
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Mapping des classes sauvegardé: {mapping_file}")
    
    return mapping_file


# ============================================================================
# 8. DÉTECTION ET CONFIGURATION DU DEVICE
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
    
    print(f"\n{'='*70}")
    print(f"🖥️  DÉTECTION DU MATÉRIEL")
    print(f"{'='*70}")
    
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {platform.python_version()}")
    print(f"   PyTorch: {torch.__version__}")
    
    if requested_device == "auto":
        device = get_available_device()
        print(f"   Device demandé: auto → {device}")
    elif requested_device in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        device = f"cuda:{requested_device}"
        print(f"   Device demandé: GPU {requested_device}")
    else:
        device = requested_device
        print(f"   Device demandé: {device}")
    
    Config.DEVICE = device
    
    if device.startswith("cuda"):
        _configure_cuda(device)
    elif device == "mps":
        _configure_mps()
    else:
        _configure_cpu()


def _configure_cuda(device: str):
    # ====== OPTIMISATIONS CUDA SÛRES ======
    # cuDNN benchmark - trouve les meilleurs algorithmes convolutionnels
    torch.backends.cudnn.benchmark = True
    
    # TF32 pour GPUs Ampere+ (H200, A100, RTX 30xx+)
    # Accélère les calculs matriciels avec précision suffisante
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"   ⚡ Optimisations CUDA: cuDNN benchmark + TF32 activés")
    # ======================================
    """Configuration pour GPU NVIDIA"""
    gpu_id = 0 if device == "cuda" else int(device.split(":")[1])
    gpu_name = torch.cuda.get_device_name(gpu_id)
    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
    
    print(f"\n   🟢 CUDA DISPONIBLE")
    print(f"   GPU: {gpu_name}")
    print(f"   Mémoire: {gpu_memory:.1f} Go")
    
    # NVIDIA H200 SXM (141 GB HBM3e)
    if "H200" in gpu_name:
        Config.BATCH_SIZE = 32
        Config.GRAD_ACCUM_STEPS = 1
        Config.MODEL_SIZE = "large"
        Config.RESOLUTION = 640
        Config.NUM_WORKERS = 16
        print(f"   → Configuration H200 SXM (141 GB)")
    
    # NVIDIA H100 (80 GB)
    elif "H100" in gpu_name:
        Config.BATCH_SIZE = 16
        Config.GRAD_ACCUM_STEPS = 1
        Config.MODEL_SIZE = "large"
        Config.RESOLUTION = 640
        Config.NUM_WORKERS = 12
        print(f"   → Configuration H100 (80 GB)")
    
    # NVIDIA A100 (40/80 GB)
    elif "A100" in gpu_name:
        if gpu_memory > 70:
            Config.BATCH_SIZE = 16
            Config.GRAD_ACCUM_STEPS = 1
        else:
            Config.BATCH_SIZE = 8
            Config.GRAD_ACCUM_STEPS = 2
        Config.MODEL_SIZE = "large"
        Config.RESOLUTION = 640
        Config.NUM_WORKERS = 12
        print(f"   → Configuration A100 ({int(gpu_memory)} GB)")
    
    # NVIDIA RTX 5090 (32 GB GDDR7)
    elif "5090" in gpu_name:
        Config.BATCH_SIZE = 16
        Config.GRAD_ACCUM_STEPS = 1
        Config.MODEL_SIZE = "large"
        Config.RESOLUTION = 640
        Config.NUM_WORKERS = 12
        print(f"   → Configuration RTX 5090 (32 GB GDDR7)")
    
    # NVIDIA RTX 4090 (24 GB)
    elif "4090" in gpu_name:
        Config.BATCH_SIZE = 8
        Config.GRAD_ACCUM_STEPS = 2
        Config.MODEL_SIZE = "base"
        Config.RESOLUTION = 640
        Config.NUM_WORKERS = 8
        print(f"   → Configuration RTX 4090 (24 GB)")
    
    # NVIDIA RTX 5080 (16 GB GDDR7)
    elif "5080" in gpu_name:
        Config.BATCH_SIZE = 8
        Config.GRAD_ACCUM_STEPS = 2
        Config.MODEL_SIZE = "base"
        Config.RESOLUTION = 640
        Config.NUM_WORKERS = 8
        print(f"   → Configuration RTX 5080 (16 GB GDDR7)")
    
    # NVIDIA RTX 3090/3090 Ti (24 GB)
    elif "3090" in gpu_name:
        Config.BATCH_SIZE = 4
        Config.GRAD_ACCUM_STEPS = 4
        Config.MODEL_SIZE = "base"
        Config.RESOLUTION = 640
        Config.NUM_WORKERS = 8
        print(f"   → Configuration RTX 3090 (24 GB)")
    
    # NVIDIA RTX 4080 (16 GB)
    elif "4080" in gpu_name:
        Config.BATCH_SIZE = 4
        Config.GRAD_ACCUM_STEPS = 4
        Config.MODEL_SIZE = "base"
        Config.RESOLUTION = 640
        Config.NUM_WORKERS = 8
        print(f"   → Configuration RTX 4080 (16 GB)")
    
    # NVIDIA RTX 3080 (10/12 GB)
    elif "3080" in gpu_name:
        Config.BATCH_SIZE = 4
        Config.GRAD_ACCUM_STEPS = 4
        Config.MODEL_SIZE = "small"
        Config.RESOLUTION = 640
        Config.NUM_WORKERS = 8
        print(f"   → Configuration RTX 3080 ({int(gpu_memory)} GB)")
    
    # Configuration générique basée sur la VRAM
    else:
        if gpu_memory >= 80:
            Config.BATCH_SIZE = 16
            Config.GRAD_ACCUM_STEPS = 1
            Config.MODEL_SIZE = "large"
            Config.RESOLUTION = 640
            Config.NUM_WORKERS = 12
            print(f"   → Configuration GPU haute capacité ({int(gpu_memory)} GB)")
        elif gpu_memory >= 32:
            Config.BATCH_SIZE = 16
            Config.GRAD_ACCUM_STEPS = 1
            Config.MODEL_SIZE = "large"
            Config.RESOLUTION = 640
            Config.NUM_WORKERS = 12
            print(f"   → Configuration GPU 32+ GB ({int(gpu_memory)} GB)")
        elif gpu_memory >= 20:
            Config.BATCH_SIZE = 8
            Config.GRAD_ACCUM_STEPS = 2
            Config.MODEL_SIZE = "base"
            Config.RESOLUTION = 640
            Config.NUM_WORKERS = 8
            print(f"   → Configuration GPU 20+ GB ({int(gpu_memory)} GB)")
        elif gpu_memory >= 12:
            Config.BATCH_SIZE = 4
            Config.GRAD_ACCUM_STEPS = 4
            Config.MODEL_SIZE = "base"
            Config.RESOLUTION = 640
            Config.NUM_WORKERS = 8
            print(f"   → Configuration GPU 12+ GB ({int(gpu_memory)} GB)")
        elif gpu_memory >= 8:
            Config.BATCH_SIZE = 2
            Config.GRAD_ACCUM_STEPS = 8
            Config.MODEL_SIZE = "small"
            Config.RESOLUTION = 560
            Config.NUM_WORKERS = 4
            print(f"   → Configuration GPU 8+ GB ({int(gpu_memory)} GB)")
        else:
            Config.BATCH_SIZE = 1
            Config.GRAD_ACCUM_STEPS = 16
            Config.MODEL_SIZE = "nano"
            Config.RESOLUTION = 480
            Config.NUM_WORKERS = 2
            print(f"   → Configuration GPU < 8 GB ({int(gpu_memory)} GB)")


def _configure_mps():
    """Configuration pour Mac Apple Silicon"""
    print(f"\n   🍎 APPLE SILICON (MPS)")
    print(f"   Chip: {platform.processor()}")
    
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print(f"   ⚠️  PYTORCH_ENABLE_MPS_FALLBACK=1 (ops non supportées → CPU)")
    
    Config.MODEL_SIZE = "nano"
    Config.BATCH_SIZE = 4
    Config.GRAD_ACCUM_STEPS = 4
    Config.RESOLUTION = 640
    Config.NUM_WORKERS = 0
    Config.LR = 1e-4
    Config.EPOCHS = 50
    Config.EARLY_STOPPING_PATIENCE = 15
    
    print(f"   → Configuration Mac (test/développement)")


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
    Config.EPOCHS = 10
    Config.EARLY_STOPPING_PATIENCE = 5


# ============================================================================
# 9. AFFICHAGE DE LA CONFIGURATION
# ============================================================================

def print_full_config_summary(dataset_name: str = None, dataset_path: Path = None, num_classes: int = None):
    """
    Affiche un GRAND résumé complet de la configuration JUSTE AVANT l'entraînement
    """
    
    # Calculs préliminaires
    effective_batch = Config.BATCH_SIZE * Config.GRAD_ACCUM_STEPS
    
    # Estimation du temps
    if Config.DEVICE.startswith("cuda"):
        gpu_id = 0 if Config.DEVICE == "cuda" else int(Config.DEVICE.split(":")[1])
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        
        if "H200" in gpu_name or "H100" in gpu_name:
            time_per_epoch = 1
        elif "5090" in gpu_name or "A100" in gpu_name:
            time_per_epoch = 2
        elif "4090" in gpu_name:
            time_per_epoch = 3
        elif "3090" in gpu_name or "5080" in gpu_name:
            time_per_epoch = 4
        else:
            time_per_epoch = 5
        
        estimated_time = f"{Config.EPOCHS * time_per_epoch} min"
        if Config.EARLY_STOPPING:
            estimated_time += f" (max, early stop activé)"
    elif Config.DEVICE == "mps":
        gpu_name = "Apple Silicon"
        gpu_memory = 0
        time_per_epoch = 10
        estimated_time = f"{Config.EPOCHS * time_per_epoch} min"
    else:
        gpu_name = "CPU"
        gpu_memory = 0
        time_per_epoch = 60
        estimated_time = f"{Config.EPOCHS * time_per_epoch} min (très lent!)"
    
    # Affichage
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "🚀 CONFIGURATION FINALE AVANT ENTRAÎNEMENT 🚀".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # SECTION 1: DEVICE / GPU
    print("║")
    print("╠" + "═" * 78 + "╣")
    print("║" + " 🖥️  DEVICE / GPU".ljust(78) + "║")
    print("╠" + "─" * 78 + "╣")
    print(f"║   Device sélectionné:     {Config.DEVICE:<50}║")
    print(f"║   GPU/Chip:               {gpu_name:<50}║")
    if gpu_memory > 0:
        print(f"║   VRAM disponible:        {gpu_memory:.1f} Go{' ' * 44}║")
    print(f"║   PyTorch version:        {torch.__version__:<50}║")
    
    # SECTION 2: MODÈLE
    print("╠" + "═" * 78 + "╣")
    print("║" + " 📦 MODÈLE RF-DETR".ljust(78) + "║")
    print("╠" + "─" * 78 + "╣")
    print(f"║   Taille du modèle:       RF-DETR-{Config.MODEL_SIZE.upper():<44}║")
    print(f"║   Résolution:             {Config.RESOLUTION} x {Config.RESOLUTION} pixels{' ' * 35}║")
    if num_classes:
        print(f"║   Nombre de classes:      {num_classes:<50}║")
    
    # SECTION 3: HYPERPARAMÈTRES
    print("╠" + "═" * 78 + "╣")
    print("║" + " 🏋️  HYPERPARAMÈTRES D'ENTRAÎNEMENT".ljust(78) + "║")
    print("╠" + "─" * 78 + "╣")
    print(f"║   Époques (max):          {Config.EPOCHS:<50}║")
    print(f"║   Batch size:             {Config.BATCH_SIZE:<50}║")
    print(f"║   Gradient accumulation:  {Config.GRAD_ACCUM_STEPS:<50}║")
    print(f"║   ➡️  Batch effectif:      {effective_batch} (batch × grad_accum){' ' * 26}║")
    print(f"║   Learning rate:          {Config.LR:<50}║")
    print(f"║   Num workers:            {Config.NUM_WORKERS:<50}║")
    
    # SECTION 4: EARLY STOPPING
    print("╠" + "═" * 78 + "╣")
    print("║" + " ⏹️  EARLY STOPPING".ljust(78) + "║")
    print("╠" + "─" * 78 + "╣")
    if Config.EARLY_STOPPING:
        print(f"║   Status:                 ✅ ACTIVÉ{' ' * 42}║")
        print(f"║   Patience:               {Config.EARLY_STOPPING_PATIENCE} époques sans amélioration{' ' * 23}║")
        print(f"║   Min delta:              {Config.EARLY_STOPPING_MIN_DELTA}{' ' * 47}║")
    else:
        print(f"║   Status:                 ❌ DÉSACTIVÉ{' ' * 39}║")
    
    # SECTION 5: DATASET
    print("╠" + "═" * 78 + "╣")
    print("║" + " 📂 DATASET".ljust(78) + "║")
    print("╠" + "─" * 78 + "╣")
    
    mode_str = "MERGED (fusion)" if Config.TRAINING_MODE == "merged" else "SINGLE (séparé)"
    print(f"║   Mode:                   {mode_str:<50}║")
    
    if dataset_name:
        print(f"║   Nom:                    {dataset_name:<50}║")
    
    if dataset_path:
        path_str = str(dataset_path)
        if len(path_str) > 48:
            path_str = "..." + path_str[-45:]
        print(f"║   Chemin:                 {path_str:<50}║")
    
    print(f"║   Fichier CSV:            {str(Config.DATASETS_CSV):<50}║")
    print(f"║   Datasets dans CSV:      {len(Config.DATASETS):<50}║")
    
    # Liste des datasets
    print("╠" + "─" * 78 + "╣")
    print("║   📋 Liste des datasets:".ljust(79) + "║")
    for name, info in list(Config.DATASETS.items())[:10]:
        url_icon = "🔗" if 'url' in info else "  "
        line = f"      {url_icon} {name}: {info['workspace']}/{info['project']} v{info['version']}"
        if len(line) > 75:
            line = line[:72] + "..."
        print(f"║{line:<78}║")
        
        if 'url' in info:
            url = info['url']
            if len(url) > 70:
                url = url[:67] + "..."
            print(f"║         ↳ {url:<67}║")
    
    if len(Config.DATASETS) > 10:
        print(f"║      ... et {len(Config.DATASETS) - 10} autres datasets{' ' * 48}║")
    
    # SECTION 6: DOSSIERS
    print("╠" + "═" * 78 + "╣")
    print("║" + " 📁 DOSSIERS".ljust(78) + "║")
    print("╠" + "─" * 78 + "╣")
    print(f"║   Base:                   {str(Config.BASE_DIR):<50}║")
    print(f"║   Datasets:               {str(Config.DATASETS_DIR):<50}║")
    print(f"║   Models:                 {str(Config.MODELS_DIR):<50}║")
    
    if Config.TRAINING_MODE == "merged":
        output_dir = Config.MODELS_DIR / Config.MERGED_MODEL_NAME
    elif dataset_name:
        output_dir = Config.MODELS_DIR / dataset_name
    else:
        output_dir = Config.MODELS_DIR / "output"
    print(f"║   Output:                 {str(output_dir):<50}║")
    
    # SECTION 7: ESTIMATION
    print("╠" + "═" * 78 + "╣")
    print("║" + " ⏱️  ESTIMATION".ljust(78) + "║")
    print("╠" + "─" * 78 + "╣")
    print(f"║   Temps par époque:       ~{time_per_epoch} min{' ' * 44}║")
    print(f"║   Temps total estimé:     ~{estimated_time:<48}║")
    
    # FOOTER
    print("║" + " " * 78 + "║")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "⚡ VÉRIFIEZ LA CONFIGURATION CI-DESSUS AVANT DE CONTINUER ⚡".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print("\n")
    
    # Pause de 3 secondes
    print("⏳ Démarrage dans 3 secondes... (Ctrl+C pour annuler)")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    print("🚀 C'est parti !\n")


# ============================================================================
# 10. FONCTIONS UTILITAIRES
# ============================================================================

def setup_directories():
    """Crée les dossiers nécessaires"""
    Config.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Dossiers créés dans {Config.BASE_DIR}")

def check_stop_file() -> bool:
    """Vérifie si l'utilisateur demande l'arrêt via fichier STOP_TRAINING"""
    stop_file = Config.BASE_DIR / "STOP_TRAINING"
    if stop_file.exists():
        print("\n🛑 Fichier STOP_TRAINING détecté - Arrêt demandé par l'utilisateur")
        return True
    return False

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

def download_dataset_coco(name: str, api_key: str) -> Optional[Path]:
    """
    Télécharge un dataset depuis Roboflow au format COCO
    
    Gère automatiquement:
    - Version non trouvée → utilise la dernière version disponible
    - Aucune version → skip le dataset avec un warning
    
    Returns:
        Path du dataset ou None si impossible à télécharger
    """
    if name not in Config.DATASETS:
        available = list(Config.DATASETS.keys())
        raise ValueError(f"Dataset inconnu: {name}. Disponibles: {available}")
    
    dataset_info = Config.DATASETS[name]
    dataset_path = Config.DATASETS_DIR / name
    
    # Déjà téléchargé ?
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"📁 Dataset '{name}' déjà présent dans {dataset_path}")
        if 'url' in dataset_info:
            print(f"   🔗 URL: {dataset_info['url']}")
        # Nettoyer les classes vides
        remove_empty_classes_from_dataset(dataset_path)
        show_dataset_stats(dataset_path)
        return dataset_path
    
    print(f"\n📥 Téléchargement du dataset '{name}' au format COCO...")
    if 'url' in dataset_info:
        print(f"   🔗 URL: {dataset_info['url']}")
    
    rf = Roboflow(api_key=api_key)
    
    try:
        project = rf.workspace(dataset_info["workspace"]).project(dataset_info["project"])
    except Exception as e:
        print(f"   ❌ Erreur accès au projet '{dataset_info['workspace']}/{dataset_info['project']}':")
        print(f"      {e}")
        print(f"   ⏭️  Dataset '{name}' ignoré")
        return None
    
    # Récupérer la version demandée
    requested_version = dataset_info["version"]
    version_to_use = None
    
    try:
        # Essayer la version demandée
        version_to_use = project.version(requested_version)
        print(f"   ✅ Version {requested_version} trouvée")
    except RuntimeError as e:
        print(f"   ⚠️  Version {requested_version} non trouvée")
        
        # Lister les versions disponibles
        try:
            # Récupérer les infos du projet pour voir les versions
            project_info = project.__dict__
            versions = []
            
            # Essayer de trouver les versions disponibles
            # Méthode 1: via l'attribut versions si disponible
            if hasattr(project, 'versions') and project.versions:
                versions = project.versions
            
            # Méthode 2: tester les versions 1 à 20
            if not versions:
                print(f"   🔍 Recherche des versions disponibles...")
                for v in range(1, 21):
                    try:
                        test_version = project.version(v)
                        versions.append(v)
                    except RuntimeError:
                        continue
            
            if versions:
                latest_version = max(versions) if isinstance(versions[0], int) else versions[-1]
                print(f"   📋 Versions disponibles: {versions}")
                print(f"   🔄 Utilisation de la version {latest_version} à la place")
                version_to_use = project.version(latest_version)
            else:
                print(f"   ❌ Aucune version disponible pour ce projet")
                print(f"   💡 Créez une version sur: https://app.roboflow.com/{dataset_info['workspace']}/{dataset_info['project']}/generate")
                print(f"   ⏭️  Dataset '{name}' ignoré")
                return None
                
        except Exception as e2:
            print(f"   ❌ Impossible de trouver une version alternative: {e2}")
            print(f"   ⏭️  Dataset '{name}' ignoré")
            return None
    
    # Télécharger
    try:
        dataset = version_to_use.download(
            model_format="coco",
            location=str(dataset_path)
        )
        print(f"   ✅ Dataset '{name}' téléchargé")
    except Exception as e:
        print(f"   ❌ Erreur lors du téléchargement: {e}")
        print(f"   ⏭️  Dataset '{name}' ignoré")
        return None
    
    # Nettoyer les classes vides
    remove_empty_classes_from_dataset(dataset_path)
    
    show_dataset_stats(dataset_path)
    
    return dataset_path

def show_dataset_stats(dataset_path: Path):
    """Affiche les statistiques du dataset"""
    print(f"\n📊 Statistiques du dataset:")
    
    categories = []
    total_images = 0
    total_annotations = 0
    
    for split in ["train", "valid", "test"]:
        split_dir = dataset_path / split
        if split_dir.exists():
            images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            
            ann_file = split_dir / "_annotations.coco.json"
            if ann_file.exists():
                with open(ann_file, 'r') as f:
                    coco = json.load(f)
                num_annotations = len(coco.get("annotations", []))
                categories = coco.get("categories", [])
                total_annotations += num_annotations
            else:
                num_annotations = "?"
            
            total_images += len(images)
            print(f"   {split:6s}: {len(images):5d} images, {num_annotations:5} annotations")
    
    if categories:
        print(f"   Classes ({len(categories)}): {[c['name'] for c in categories]}")
    
    return {"images": total_images, "annotations": total_annotations, "categories": categories}


# ============================================================================
# 11. ENTRAÎNEMENT RF-DETR
# ============================================================================

def train_rfdetr(dataset_path: Path, model_name: str, class_info: dict, epochs: int = None) -> str:
    """
    Entraîne RF-DETR sur un dataset
    """
    epochs = epochs or Config.EPOCHS
    
    # VALIDATION DU DATASET COCO
    print(f"\n{'='*70}")
    print(f"🔒 VÉRIFICATION PRÉ-ENTRAÎNEMENT")
    print(f"{'='*70}")
    
    if not ensure_valid_coco_dataset(dataset_path):
        raise ValueError(f"❌ Dataset COCO invalide et impossible à corriger: {dataset_path}")
    
    # Nombre de classes
    categories = extract_classes_from_dataset(dataset_path)
    num_classes = len(categories)
    
    # Mettre à jour class_info avec les IDs corrigés
    if not class_info.get("classes"):
        class_info = {"classes": {}}
        for cat in categories:
            class_info["classes"][cat['id']] = {
                'name': cat['name'],
                'supercategory': cat.get('supercategory', ''),
                'source_datasets': [model_name]
            }
    
    # AFFICHAGE DU RÉSUMÉ
    print_full_config_summary(
        dataset_name=model_name,
        dataset_path=dataset_path,
        num_classes=num_classes
    )
    
    # Charger le modèle
    model = get_model(Config.MODEL_SIZE)
    
    # Chemin de sortie
    output_dir = Config.MODELS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder le mapping des classes
    datasets_used = list(class_info.get("source_datasets", {}).keys()) if "source_datasets" in class_info else [model_name]
    save_class_mapping(output_dir, class_info, datasets_used)
    
    print(f"\n🏋️ Lancement de l'entraînement...")
    print(f"   Classes: {num_classes}")
    print(f"   IDs: {list(class_info['classes'].keys())}")
    
    # Construire les paramètres d'entraînement
    train_params = {
        "dataset_dir": str(dataset_path),
        "epochs": epochs,
        "batch_size": Config.BATCH_SIZE,
        "grad_accum_steps": Config.GRAD_ACCUM_STEPS,
        "lr": Config.LR,
        "output_dir": str(output_dir),
        "resolution": Config.RESOLUTION,
        "device": Config.DEVICE,
        "num_workers": Config.NUM_WORKERS,
    }
    # === WANDB ===
    # Initialiser wandb si disponible
    wandb_run = None
    try:
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "rf-detr-detection"),
            name=model_name,
            config={
                "model_size": Config.MODEL_SIZE,
                "epochs": epochs,
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LR,
                "resolution": Config.RESOLUTION,
                "num_classes": num_classes,
                "dataset": model_name,
            },
            resume="allow",
        )
        print(f"   📊 Wandb initialisé: {wandb_run.url}")
    except Exception as e:
        print(f"   ⚠️  Wandb non disponible: {e}")
        print(f"   💡 Pour activer: pip install wandb && wandb login")
    
    # Dans train_rfdetr, après la création de train_params
    if args.resume and Path(args.resume).exists():
        train_params["resume"] = args.resume
        print(f"   🔄 Reprise depuis: {args.resume}")

    # Ajouter early stopping si activé
    if Config.EARLY_STOPPING:
        train_params["early_stopping"] = True
        train_params["early_stopping_patience"] = Config.EARLY_STOPPING_PATIENCE
        train_params["early_stopping_min_delta"] = Config.EARLY_STOPPING_MIN_DELTA
    
    train_params["pin_memory"] = True
    train_params["persistent_workers"] = True
    train_params["prefetch_factor"] = 4

    # Lancer l'entraînement
    try:
        model.train(**train_params)
    except TypeError as e:
        print(f"⚠️ Certains paramètres non supportés, tentative sans early stopping...")
        print(f"   Erreur: {e}\n")
        model.train(
            dataset_dir=str(dataset_path),
            epochs=epochs,
            batch_size=Config.BATCH_SIZE,
            grad_accum_steps=Config.GRAD_ACCUM_STEPS,
            lr=Config.LR,
            output_dir=str(output_dir),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
    
    # Trouver le meilleur modèle
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
# 12. ÉVALUATION ET PRÉDICTION
# ============================================================================

def evaluate_model(model_path: str, dataset_path: Path) -> dict:
    """Évalue le modèle sur le dataset de test"""
    from PIL import Image
    import numpy as np
    
    print(f"\n{'='*70}")
    print(f"📊 ÉVALUATION DU MODÈLE")
    print(f"{'='*70}")
    
    model = get_model(Config.MODEL_SIZE)
    if model_path and Path(model_path).exists():
        model = type(model)(pretrain_weights=str(model_path))
        print(f"   ✅ Modèle chargé: {model_path}")
    
    mapping_file = Path(model_path).parent / "class_mapping.json"
    class_names = {}
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        for class_id, info in mapping.get('classes', {}).items():
            class_names[int(class_id)] = info['name']
    
    test_dir = dataset_path / "test"
    if not test_dir.exists():
        test_dir = dataset_path / "valid"
    
    ann_file = test_dir / "_annotations.coco.json"
    if not ann_file.exists():
        print(f"   ❌ Pas de fichier d'annotations de test")
        return {}
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    stats = {cat['id']: {
        'name': cat['name'],
        'total_gt': 0,
        'total_pred': 0,
        'correct': 0,
        'confidences': []
    } for cat in coco_data['categories']}
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
        stats[ann['category_id']]['total_gt'] += 1
    
    print(f"   📷 Évaluation sur {len(coco_data['images'])} images...")
    
    for img_info in coco_data['images']:
        img_path = test_dir / img_info['file_name']
        if not img_path.exists():
            continue
        
        image = Image.open(img_path)
        detections = model.predict(image, threshold=0.5)
        
        if len(detections) > 0 and hasattr(detections, 'class_id'):
            for cls_id, conf in zip(detections.class_id, detections.confidence):
                if cls_id in stats:
                    stats[cls_id]['total_pred'] += 1
                    stats[cls_id]['confidences'].append(float(conf))
    
    print(f"\n   {'─'*60}")
    print(f"   {'Classe':<25} {'GT':>8} {'Pred':>8} {'Conf Moy':>12}")
    print(f"   {'─'*60}")
    
    for cls_id, s in stats.items():
        avg_conf = np.mean(s['confidences']) if s['confidences'] else 0
        print(f"   {s['name']:<25} {s['total_gt']:>8} {s['total_pred']:>8} {avg_conf:>12.1%}")
    
    print(f"   {'─'*60}")
    
    return stats


def predict_test(model_path: str = None, image_path: str = None, save_dir: str = None):
    """Test de prédiction rapide"""
    from PIL import Image
    import requests
    from io import BytesIO
    
    print(f"\n{'='*70}")
    print(f"🔍 TEST DE PRÉDICTION")
    print(f"{'='*70}")
    
    model = get_model(Config.MODEL_SIZE)
    
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
    print(f"   🎯 Résolution d'inférence: {Config.RESOLUTION} px")
    
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


# ============================================================================
# 13. FONCTIONS PRINCIPALES
# ============================================================================

def train_single_dataset(dataset_name: str):
    """Pipeline pour un seul dataset avec gestion interactive des classes"""
    setup_directories()
    
    # Télécharger le dataset
    dataset_path = download_dataset_coco(dataset_name, Config.ROBOFLOW_API_KEY)
    
    # Gestion interactive des classes
    class_info = interactive_class_management(dataset_path, dataset_name)
    
    # Entraîner
    model_path = train_rfdetr(dataset_path, dataset_name, class_info)
    
    # Évaluer
    try:
        evaluate_model(model_path, dataset_path)
    except Exception as e:
        print(f"⚠️ Erreur évaluation: {e}")
    
    # Test de prédiction
    try:
        predict_test(model_path)
    except Exception as e:
        print(f"⚠️ Erreur prédiction test: {e}")
    
    return model_path

def train_merged_datasets(dataset_names: List[str] = None):
    """Pipeline pour fusionner et entraîner sur plusieurs datasets"""
    setup_directories()
    
    if dataset_names is None:
        dataset_names = list(Config.DATASETS.keys())
    
    print(f"\n{'='*70}")
    print(f"🔀 MODE FUSION - {len(dataset_names)} datasets")
    print(f"{'='*70}")
    print(f"   Datasets: {', '.join(dataset_names)}")
    
    # Télécharger tous les datasets
    dataset_paths = []
    successful_names = []
    failed_names = []
    
    for name in dataset_names:
        path = download_dataset_coco(name, Config.ROBOFLOW_API_KEY)
        if path is not None:
            dataset_paths.append(path)
            successful_names.append(name)
        else:
            failed_names.append(name)
    
    # Vérifier qu'on a au moins un dataset
    if not dataset_paths:
        print(f"\n❌ Aucun dataset n'a pu être téléchargé!")
        print(f"   Vérifiez votre fichier CSV et les versions des projets Roboflow")
        return None
    
    # Résumé des téléchargements
    if failed_names:
        print(f"\n{'─'*60}")
        print(f"   ⚠️  {len(failed_names)} dataset(s) ignoré(s): {', '.join(failed_names)}")
        print(f"   ✅ {len(successful_names)} dataset(s) disponible(s): {', '.join(successful_names)}")
        print(f"{'─'*60}")
    
    # Analyser toutes les classes (avec les listes filtrées!)
    if Config.INTERACTIVE_MODE:
        all_stats = analyze_all_datasets_classes(dataset_paths, successful_names)
        display_merged_classes_table(all_stats)
        merge_groups, renames = interactive_merge_classes_merged_mode(all_stats)
    else:
        merge_groups = []
        renames = None
    
    # Fusionner les datasets
    merged_path = Config.DATASETS_DIR / Config.MERGED_MODEL_NAME
    class_info = merge_coco_datasets(dataset_paths, merged_path, merge_groups, renames)
    
    # Afficher les stats du dataset fusionné
    show_dataset_stats(merged_path)
    
    # Entraîner sur le dataset fusionné
    model_path = train_rfdetr(merged_path, Config.MERGED_MODEL_NAME, class_info)
    
    # Évaluer
    try:
        evaluate_model(model_path, merged_path)
    except Exception as e:
        print(f"⚠️ Erreur évaluation: {e}")
    
    # Test de prédiction
    try:
        predict_test(model_path)
    except Exception as e:
        print(f"⚠️ Erreur prédiction test: {e}")
    
    return model_path

def train_all_separate():
    """Entraîne un modèle séparé pour chaque dataset"""
    models = {}
    setup_directories()
    
    for dataset_name in Config.DATASETS.keys():
        try:
            print(f"\n\n{'#'*70}")
            print(f"# DATASET: {dataset_name.upper()}")
            print(f"{'#'*70}")
            
            model_path = train_single_dataset(dataset_name)
            models[dataset_name] = model_path
            
        except Exception as e:
            print(f"❌ Erreur pour {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"📋 RÉSUMÉ FINAL")
    print(f"{'='*70}")
    for name, path in models.items():
        print(f"   ✅ {name}: {path}")
    
    return models


# ============================================================================
# 14. POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Entraînement RF-DETR - Compatible CUDA, MPS (Mac), CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Mode single: un modèle par dataset
  python train_rfdetr.py --api-key CLE --dataset sanglier
  
  # Mode merged: fusion de tous les datasets → un seul modèle
  python train_rfdetr.py --api-key CLE --mode merged
  
  # Mode merged avec datasets spécifiques
  python train_rfdetr.py --api-key CLE --mode merged --datasets sanglier,frelon
  
  # Configuration personnalisée
  python train_rfdetr.py --api-key CLE --mode merged --resolution 640 --epochs 200
  
  # Mode non-interactif (pas de questions)
  python train_rfdetr.py --api-key CLE --mode merged --no-interactive

Format du fichier CSV (datasets.csv):
  name,workspace,project,version,url
  sanglier,mon-workspace,mon-projet,1,https://app.roboflow.com/...

Modes d'entraînement:
  single  : Un modèle par dataset (défaut)
  merged  : Fusion des datasets → Un seul modèle multi-classes
        """
    )
    
    # Arguments principaux
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset spécifique (mode single) ou 'all'")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Liste de datasets séparés par virgule (mode merged)")
    parser.add_argument("--datasets-csv", type=str, default="datasets.csv",
                        help="Chemin vers le fichier CSV des datasets")
    parser.add_argument("--mode", type=str, choices=["single", "merged"], default="single",
                        help="Mode: 'single' ou 'merged'")
    parser.add_argument("--model-name", type=str, default="merged_model",
                        help="Nom du modèle fusionné (mode merged)")
    parser.add_argument("--api-key", type=str, required=False,
                        help="Clé API Roboflow")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'cuda:0', 'mps', 'cpu'")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Nombre d'époques max (défaut: 200)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Taille du batch")
    parser.add_argument("--model-size", type=str, choices=["nano", "small", "base", "large"],
                        default=None, help="Taille du modèle")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Résolution des images (défaut: 640)")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Gradient accumulation steps")
    parser.add_argument("--workers", type=int, default=None,
                        help="Nombre de workers")
    
    # Early stopping
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Désactiver l'early stopping")
    parser.add_argument("--patience", type=int, default=None,
                        help="Patience pour early stopping (défaut: 50)")
    
    # Mode interactif
    parser.add_argument("--no-interactive", action="store_true",
                        help="Désactiver le mode interactif (pas de questions)")
    
    # Modes spéciaux
    parser.add_argument("--predict", action="store_true",
                        help="Mode prédiction")
    parser.add_argument("--evaluate", action="store_true",
                        help="Mode évaluation")
    parser.add_argument("--model", type=str,
                        help="Chemin vers le modèle")
    parser.add_argument("--image", type=str,
                        help="Image pour prédiction")
    parser.add_argument("--list-datasets", action="store_true",
                        help="Liste les datasets avec URLs")
    parser.add_argument("--show-config", action="store_true",
                        help="Affiche la configuration sans lancer l'entraînement")
    parser.add_argument("--validate-only", action="store_true",
                        help="Valide/corrige le dataset sans entraîner")
    parser.add_argument("--resume", type=str, default=None,
                        help="Chemin vers checkpoint pour reprendre l'entraînement")
    args = parser.parse_args()
    
    # Charger les datasets depuis le CSV
    Config.DATASETS_CSV = Path(args.datasets_csv)
    Config.TRAINING_MODE = args.mode
    Config.MERGED_MODEL_NAME = args.model_name
    Config.INTERACTIVE_MODE = not args.no_interactive
    
    try:
        Config.DATASETS = load_datasets_from_csv(Config.DATASETS_CSV)
    except FileNotFoundError:
        if not args.list_datasets:
            print(f"\n💡 Créez le fichier {args.datasets_csv} avec le format:")
            print(f"   name,workspace,project,version,url")
            print(f"   sanglier,mon-workspace,mon-projet,1,https://app.roboflow.com/...")
            exit(1)
    
    # Mode liste des datasets avec URLs
    if args.list_datasets:
        print(f"\n{'='*70}")
        print(f"📋 DATASETS DISPONIBLES ({Config.DATASETS_CSV})")
        print(f"{'='*70}")
        for name, info in Config.DATASETS.items():
            print(f"\n   📦 {name}")
            print(f"      Workspace: {info['workspace']}")
            print(f"      Project:   {info['project']}")
            print(f"      Version:   {info['version']}")
            if 'url' in info:
                print(f"      🔗 URL:    {info['url']}")
            if 'notes' in info:
                print(f"      📝 Notes:  {info['notes']}")
        print(f"\n{'='*70}")
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
    if args.no_early_stopping:
        Config.EARLY_STOPPING = False
    if args.patience is not None:
        Config.EARLY_STOPPING_PATIENCE = args.patience
    
    # Mode affichage config seulement
    if args.show_config:
        print_full_config_summary()
        exit(0)
    
    # Mode validation seulement
    if args.validate_only:
        setup_directories()
        if not Config.ROBOFLOW_API_KEY:
            Config.ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
        if not Config.ROBOFLOW_API_KEY:
            print("❌ Clé API Roboflow requise pour télécharger les datasets")
            exit(1)
        
        if args.dataset:
            dataset_path = download_dataset_coco(args.dataset, Config.ROBOFLOW_API_KEY)
            # En mode validate-only, on affiche aussi les stats des classes
            class_stats = analyze_dataset_classes(dataset_path)
            display_classes_table(args.dataset, class_stats)
            ensure_valid_coco_dataset(dataset_path)
        else:
            for name in Config.DATASETS.keys():
                print(f"\n{'='*70}")
                print(f"📦 Dataset: {name}")
                print(f"{'='*70}")
                dataset_path = download_dataset_coco(name, Config.ROBOFLOW_API_KEY)
                class_stats = analyze_dataset_classes(dataset_path)
                display_classes_table(name, class_stats)
                ensure_valid_coco_dataset(dataset_path)
        exit(0)
    
    # Mode prédiction
    if args.predict:
        setup_directories()
        predict_test(args.model, args.image)
        exit(0)
    
    # Mode évaluation
    if args.evaluate:
        setup_directories()
        if not args.model:
            print("❌ --model requis pour l'évaluation")
            exit(1)
        model_dir = Path(args.model).parent
        dataset_path = Config.DATASETS_DIR / model_dir.name
        if not dataset_path.exists():
            dataset_path = Config.DATASETS_DIR / Config.MERGED_MODEL_NAME
        evaluate_model(args.model, dataset_path)
        exit(0)
    
    # Vérifier la clé API
    if not Config.ROBOFLOW_API_KEY:
        Config.ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
    
    if not Config.ROBOFLOW_API_KEY:
        print("❌ Clé API Roboflow requise")
        print("   --api-key VOTRE_CLE")
        print("   ou: export ROBOFLOW_API_KEY=VOTRE_CLE")
        exit(1)
    
    # Bannière de démarrage
    device_emoji = "🖥️" if Config.DEVICE.startswith("cuda") else "🍎" if Config.DEVICE == "mps" else "💻"
    mode_emoji = "🔀" if args.mode == "merged" else "📦"
    interactive_emoji = "💬" if Config.INTERACTIVE_MODE else "🤖"
    
    print("\n")
    print("🐗" * 35)
    print("🐗" + " " * 66 + "🐗")
    print("🐗" + "   RF-DETR TRAINING PIPELINE".center(66) + "🐗")
    print("🐗" + f"   {device_emoji} Device: {Config.DEVICE}".center(66) + "🐗")
    print("🐗" + f"   {mode_emoji} Mode: {args.mode}".center(66) + "🐗")
    print("🐗" + f"   {interactive_emoji} Interactif: {'Oui' if Config.INTERACTIVE_MODE else 'Non'}".center(66) + "🐗")
    print("🐗" + f"   🎯 Resolution: {Config.RESOLUTION}px".center(66) + "🐗")
    print("🐗" + f"   ⏱️  Epochs: {Config.EPOCHS} (patience: {Config.EARLY_STOPPING_PATIENCE})".center(66) + "🐗")
    print("🐗" + " " * 66 + "🐗")
    print("🐗" * 35)
    print("\n")
    
    # Lancer l'entraînement selon le mode
    if args.mode == "merged":
        if args.datasets:
            dataset_list = [d.strip() for d in args.datasets.split(",")]
            for d in dataset_list:
                if d not in Config.DATASETS:
                    print(f"❌ Dataset '{d}' non trouvé dans {Config.DATASETS_CSV}")
                    exit(1)
            train_merged_datasets(dataset_list)
        else:
            train_merged_datasets()
    else:
        if args.dataset == "all":
            train_all_separate()
        elif args.dataset:
            if args.dataset not in Config.DATASETS:
                print(f"❌ Dataset '{args.dataset}' non trouvé")
                print(f"   Disponibles: {list(Config.DATASETS.keys())}")
                exit(1)
            train_single_dataset(args.dataset)
        else:
            if len(Config.DATASETS) == 1:
                train_single_dataset(list(Config.DATASETS.keys())[0])
            else:
                print(f"❌ Spécifiez --dataset ou utilisez --mode merged")
                print(f"   Disponibles: {list(Config.DATASETS.keys())}")
                exit(1)
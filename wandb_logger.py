import re
import sys
import wandb

wandb.init(project="rf-detr-detection", name="nano")

current_epoch = 0
step = 0

for line in sys.stdin:
    print(line, end='')  # Affiche la ligne normalement
    
    try:
        # ============================================================
        # TRAINING LOGS
        # ============================================================
        if "Epoch:" in line and "loss:" in line and "Test:" not in line:
            epoch_match = re.search(r'Epoch: \[(\d+)\]', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            metrics = {'epoch': current_epoch}
            patterns = {
                'train/loss': r'loss: [\d.]+ \(([\d.]+)\)',
                'train/loss_ce': r'loss_ce: [\d.]+ \(([\d.]+)\)',
                'train/loss_bbox': r'loss_bbox: [\d.]+ \(([\d.]+)\)',
                'train/loss_giou': r'loss_giou: [\d.]+ \(([\d.]+)\)',
                'train/class_error': r'class_error: ([\d.]+)',
                'train/lr': r'lr: ([\d.]+)',
            }
            
            for name, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    metrics[name] = float(match.group(1))
            
            if len(metrics) > 1:
                wandb.log(metrics, step=step)
                step += 1
        
        # ============================================================
        # VALIDATION/TEST LOSS LOGS
        # ============================================================
        elif "Evaluation results:" in line:
            metrics = {'epoch': current_epoch}
            patterns = {
                'val/loss': r'loss: [\d.]+ \(([\d.]+)\)',
                'val/loss_ce': r'loss_ce: [\d.]+ \(([\d.]+)\)',
                'val/loss_bbox': r'loss_bbox: [\d.]+ \(([\d.]+)\)',
                'val/loss_giou': r'loss_giou: [\d.]+ \(([\d.]+)\)',
                'val/class_error': r'class_error: ([\d.]+)',
            }
            
            for name, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    metrics[name] = float(match.group(1))
            
            if len(metrics) > 1:
                wandb.log(metrics, step=step)
        
        # ============================================================
        # mAP METRICS (les plus importantes!)
        # ============================================================
        elif "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all" in line:
            match = re.search(r'= ([\d.]+)', line)
            if match:
                wandb.log({'val/mAP_50_95': float(match.group(1)), 'epoch': current_epoch}, step=step)
        
        elif "Average Precision  (AP) @[ IoU=0.50      | area=   all" in line:
            match = re.search(r'= ([\d.]+)', line)
            if match:
                wandb.log({'val/mAP_50': float(match.group(1)), 'epoch': current_epoch}, step=step)
        
        elif "Average Precision  (AP) @[ IoU=0.75      | area=   all" in line:
            match = re.search(r'= ([\d.]+)', line)
            if match:
                wandb.log({'val/mAP_75': float(match.group(1)), 'epoch': current_epoch}, step=step)
        
        # AP par taille d'objet
        elif "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small" in line:
            match = re.search(r'= ([\d.]+)', line)
            if match:
                wandb.log({'val/AP_small': float(match.group(1)), 'epoch': current_epoch}, step=step)
        
        elif "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium" in line:
            match = re.search(r'= ([\d.]+)', line)
            if match:
                wandb.log({'val/AP_medium': float(match.group(1)), 'epoch': current_epoch}, step=step)
        
        elif "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large" in line:
            match = re.search(r'= ([\d.]+)', line)
            if match:
                wandb.log({'val/AP_large': float(match.group(1)), 'epoch': current_epoch}, step=step)
        
        # ============================================================
        # RECALL METRICS
        # ============================================================
        elif "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=500" in line:
            match = re.search(r'= ([\d.]+)', line)
            if match:
                wandb.log({'val/AR_500': float(match.group(1)), 'epoch': current_epoch}, step=step)
        
        # ============================================================
        # PER-CLASS METRICS (très utile!)
        # ============================================================
        elif "'class':" in line and "map@50:95" in line:
            # Parser les résultats par classe
            class_match = re.search(r"'class': '(\w+)'", line)
            map_match = re.search(r"'map@50:95': ([\d.]+)", line)
            map50_match = re.search(r"'map@50': ([\d.]+)", line)
            precision_match = re.search(r"'precision': [\w.()]+\(([\d.]+)\)", line)
            recall_match = re.search(r"'recall': [\w.()]+\(([\d.]+)\)", line)
            f1_match = re.search(r"'f1_score': [\w.()]+\(([\d.]+)\)", line)
            
            if class_match and class_match.group(1) != 'all':
                class_name = class_match.group(1)
                class_metrics = {'epoch': current_epoch}
                
                if map_match:
                    class_metrics[f'class/{class_name}/mAP_50_95'] = float(map_match.group(1))
                if map50_match:
                    class_metrics[f'class/{class_name}/mAP_50'] = float(map50_match.group(1))
                if precision_match:
                    class_metrics[f'class/{class_name}/precision'] = float(precision_match.group(1))
                if recall_match:
                    class_metrics[f'class/{class_name}/recall'] = float(recall_match.group(1))
                if f1_match:
                    class_metrics[f'class/{class_name}/f1'] = float(f1_match.group(1))
                
                if len(class_metrics) > 1:
                    wandb.log(class_metrics, step=step)
        
        # ============================================================
        # OVERALL METRICS (classe 'all')
        # ============================================================
        elif "'class': 'all'" in line:
            precision_match = re.search(r"'precision': [\w.()]+\(([\d.]+)\)", line)
            recall_match = re.search(r"'recall': [\w.()]+\(([\d.]+)\)", line)
            f1_match = re.search(r"'f1_score': [\w.()]+\(([\d.]+)\)", line)
            
            overall_metrics = {'epoch': current_epoch}
            if precision_match:
                overall_metrics['val/precision'] = float(precision_match.group(1))
            if recall_match:
                overall_metrics['val/recall'] = float(recall_match.group(1))
            if f1_match:
                overall_metrics['val/f1_score'] = float(f1_match.group(1))
            
            if len(overall_metrics) > 1:
                wandb.log(overall_metrics, step=step)
    
    except Exception as e:
        # Silencieux pour ne pas interrompre le training
        pass

wandb.finish()
#!/usr/bin/env python
# download_models.py
# Télécharge les 3 modèles requis pour le pipeline v2.0 :
#   - yolov9c.pt         (détection joueurs/ballon/arbitres)
#   - yolov8n-pose.pt    (estimation de pose, 17 keypoints COCO)
#   - ResNet-18          (classification d'équipe, via torchvision)

import sys
import os
from config.settings import configure_runtime_environment, get_model_device

configure_runtime_environment()
DEVICE = get_model_device()

def banner(titre):
    print(f"\n{'='*60}")
    print(f"  {titre}")
    print(f"{'='*60}")

# ── 1. YOLOv9c ────────────────────────────────────────────────
banner("1/3  YOLOv9c  (yolov9c.pt)")
try:
    from ultralytics import YOLO
    model = YOLO("yolov9c.pt")   # télécharge dans ~/.config/Ultralytics/ ou CWD
    model.to(DEVICE)
    # Vérification rapide
    info = model.info(verbose=False)
    dest = model.ckpt_path if hasattr(model, 'ckpt_path') else "yolov9c.pt"
    print(f"[OK] YOLOv9c téléchargé → {dest}")
except Exception as e:
    print(f"[ERREUR] YOLOv9c : {e}", file=sys.stderr)

# ── 2. YOLOv8n-Pose ───────────────────────────────────────────
banner("2/3  YOLOv8-Pose  (yolov8n-pose.pt)")
try:
    from ultralytics import YOLO
    model_pose = YOLO("yolov8n-pose.pt")
    model_pose.to(DEVICE)
    info_pose = model_pose.info(verbose=False)
    dest_pose = model_pose.ckpt_path if hasattr(model_pose, 'ckpt_path') else "yolov8n-pose.pt"
    print(f"[OK] YOLOv8n-Pose téléchargé → {dest_pose}")
except Exception as e:
    print(f"[ERREUR] YOLOv8n-Pose : {e}", file=sys.stderr)

# ── 3. ResNet-18 (torchvision) ────────────────────────────────
banner("3/3  ResNet-18  (torchvision — ImageNet weights)")
try:
    import torch
    import torchvision.models as models
    from torchvision.models import ResNet18_Weights

    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.to(DEVICE)
    resnet.eval()

    # Localisation du cache
    cache_dir = torch.hub.get_dir()
    print(f"[OK] ResNet-18 téléchargé → cache : {cache_dir}")

    # Vérification rapide : forward sur un tenseur fictif
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=DEVICE)
        out = resnet(dummy)
    print(f"     Sortie forward OK : shape {tuple(out.shape)}")
except Exception as e:
    print(f"[ERREUR] ResNet-18 : {e}", file=sys.stderr)

# ── Résumé ────────────────────────────────────────────────────
banner("Téléchargements terminés")
print("Les modèles sont prêts pour le pipeline v2.0.")
print(f"  - Device            : {DEVICE}")
print(f"  - Ultralytics cache : {os.environ.get('YOLO_CONFIG_DIR', 'default')}")
print(f"  - Torch hub cache   : {os.environ.get('TORCH_HOME', 'default')}")
print()

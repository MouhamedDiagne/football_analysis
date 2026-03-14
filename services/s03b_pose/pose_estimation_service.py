# services/s03b_pose/pose_estimation_service.py  — v2.0
# Estimation de pose corporelle (17 keypoints COCO) via YOLOv8-Pose

import numpy as np
from typing import Dict, List, Optional, Tuple
from config.settings import get_model_device
from core.base_service import BaseService
from core.data_models import Joueur, PoseKeypoints, BoundingBox


class PoseEstimationService(BaseService):
    """
    Estimation de pose avec YOLOv8n-Pose.
    Produit 17 keypoints COCO par joueur détecté.
    Association aux tracks via IoU de bounding box.
    """

    def __init__(self,
                 model_path: str = "yolov8n-pose.pt",
                 confidence: float = 0.35):
        super().__init__("PoseEstimation")
        self.model_path = model_path
        self.confidence = confidence
        self.model      = None
        self.device     = get_model_device()
        self._stats = {'frames': 0, 'poses_detectees': 0, 'poses_associees': 0}

    def initialiser(self):
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        try:
            self.model.to(self.device)
        except Exception:
            self.logger.warning(f"Device {self.device} non force sur YOLO Pose, fallback implicite.")
        self.logger.info(f"   Device Pose: {self.device}")
        self.logger.info(f"✅ YOLOv8-Pose chargé : {self.model_path}")
        self.est_initialise = True

    def traiter(self, data: dict) -> Dict[int, PoseKeypoints]:
        return self.estimer_poses(
            data['frame'],
            data['joueurs'],
            data['frame_id']
        )

    # ─────────────────────────────────────
    # ESTIMATION PRINCIPALE
    # ─────────────────────────────────────

    def estimer_poses(self,
                      frame: np.ndarray,
                      joueurs: List[Joueur],
                      frame_id: int) -> Dict[int, PoseKeypoints]:
        """
        Estime les poses dans la frame, les associe aux joueurs trackés
        et injecte le résultat dans joueur.pose.

        Retourne : Dict[joueur_id → PoseKeypoints]
        """
        if not joueurs:
            return {}

        results = self.model(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )[0]

        poses_raw = self._extraire_poses_raw(results, frame_id)
        self._stats['frames']          += 1
        self._stats['poses_detectees'] += len(poses_raw)

        poses_associees = self._associer_aux_joueurs(joueurs, poses_raw)
        self._stats['poses_associees'] += len(poses_associees)

        # Injecter la pose dans chaque objet Joueur
        for jid, pose in poses_associees.items():
            for j in joueurs:
                if j.id == jid:
                    j.pose = pose
                    break

        return poses_associees

    # ─────────────────────────────────────
    # EXTRACTION BRUTE
    # ─────────────────────────────────────

    def _extraire_poses_raw(self,
                             results,
                             frame_id: int
                             ) -> List[Tuple[BoundingBox, PoseKeypoints]]:
        """Extrait les (BoundingBox, PoseKeypoints) bruts du résultat YOLO."""
        poses = []

        if results.keypoints is None or results.boxes is None:
            return poses

        kpts_data  = results.keypoints.data   # (N, 17, 3) : x, y, conf
        boxes_data = results.boxes.xyxy        # (N, 4)

        for i in range(len(kpts_data)):
            kp  = kpts_data[i].cpu().numpy()   # (17, 3)
            box = boxes_data[i].cpu().numpy()   # (4,)

            keypoints   = kp[:, :2]   # (17, 2)
            confidences = kp[:, 2]    # (17,)
            x1, y1, x2, y2 = map(int, box)

            bbox = BoundingBox(x1, y1, x2, y2)
            pose = PoseKeypoints(
                keypoints=keypoints,
                confidences=confidences,
                frame_id=frame_id
            )
            poses.append((bbox, pose))

        return poses

    # ─────────────────────────────────────
    # ASSOCIATION AUX JOUEURS TRACKÉS
    # ─────────────────────────────────────

    def _associer_aux_joueurs(self,
                               joueurs: List[Joueur],
                               poses_raw: List[Tuple[BoundingBox, PoseKeypoints]]
                               ) -> Dict[int, PoseKeypoints]:
        """
        Associe chaque pose détectée au joueur avec la meilleure IoU.
        Un joueur ne peut être associé qu'à une seule pose.
        """
        associees: Dict[int, PoseKeypoints] = {}
        utilise = [False] * len(poses_raw)

        # Trier les joueurs par confiance décroissante pour prioritiser
        joueurs_tries = sorted(joueurs,
                               key=lambda j: j.confiance_detection,
                               reverse=True)

        for joueur in joueurs_tries:
            meilleur_iou = 0.3   # seuil minimum IoU pour association
            meilleur_idx = -1

            for idx, (pose_bbox, _) in enumerate(poses_raw):
                if utilise[idx]:
                    continue
                iou = joueur.bbox.iou(pose_bbox)
                if iou > meilleur_iou:
                    meilleur_iou = iou
                    meilleur_idx = idx

            if meilleur_idx >= 0:
                _, pose = poses_raw[meilleur_idx]
                associees[joueur.id] = pose
                utilise[meilleur_idx] = True

        return associees

    def obtenir_stats(self) -> dict:
        return {**self._stats}

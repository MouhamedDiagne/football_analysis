# services/s03_detection/player_detection_service.py  — v2.0
# Détection avec YOLOv9c : joueurs, ballon, arbitres

import cv2
import numpy as np
from typing import List, Optional
from config.settings import get_model_device
from core.base_service import BaseService
from core.data_models import Joueur, Ballon, Arbitre, BoundingBox, Position, RoleDetecte

# Classes YOLO COCO
CLASSE_PERSONNE = 0
CLASSE_BALLON   = 32

# Plages HSV pour détecter le maillot d'arbitre (jaune fluo)
ARBITRE_HSV_JAUNE_BAS = np.array([18, 100, 100])
ARBITRE_HSV_JAUNE_HT  = np.array([35, 255, 255])


class PlayerDetectionService(BaseService):
    """
    Détection multi-classes via YOLOv9c.
    Sorties : joueurs, arbitres, ballon.
    """

    def __init__(self,
                 model_path: str = "yolov9c.pt",
                 confidence: float = 0.35,
                 iou_threshold: float = 0.45):
        super().__init__("Detection")
        self.model_path    = model_path
        self.confidence    = confidence
        self.iou_threshold = iou_threshold
        self.model         = None
        self.device        = get_model_device()
        self._stats = {
            'frames':   0,
            'joueurs':  0,
            'ballons':  0,
            'arbitres': 0,
            'conf_moy': 0.0,
        }

    def initialiser(self):
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        try:
            self.model.to(self.device)
        except Exception:
            self.logger.warning(f"Device {self.device} non force sur YOLO, fallback implicite.")
        self.logger.info(f"   Device YOLO: {self.device}")
        self.logger.info(f"✅ YOLOv9c chargé : {self.model_path}")
        self.est_initialise = True

    def traiter(self, data: dict) -> dict:
        return self.detecter(data['frame'], data['frame_id'], data['timestamp'])

    # ─────────────────────────────────────
    # DÉTECTION PRINCIPALE
    # ─────────────────────────────────────

    def detecter(self,
                 frame: np.ndarray,
                 frame_id: int,
                 timestamp: float) -> dict:
        """
        Détecte joueurs, arbitres et ballon dans une frame.

        Retourne :
            {
              'joueurs' : List[Joueur],
              'arbitres': List[Arbitre],
              'ballon'  : Optional[Ballon],
            }
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=[CLASSE_PERSONNE, CLASSE_BALLON],
            device=self.device,
            verbose=False
        )[0]

        joueurs:  List[Joueur]  = []
        arbitres: List[Arbitre] = []
        ballon:   Optional[Ballon] = None
        meilleure_conf_ballon = 0.0
        frame_h, frame_w = frame.shape[:2]
        confs_frame = []

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bbox = BoundingBox(x1, y1, x2, y2)
            pos  = Position(cx, cy, timestamp, frame_id)
            confs_frame.append(conf)

            if cls == CLASSE_BALLON:
                if conf > meilleure_conf_ballon:
                    meilleure_conf_ballon = conf
                    ballon = Ballon(position=pos, bbox=bbox, confiance=conf)

            elif cls == CLASSE_PERSONNE:
                if not self._est_joueur_valide(bbox, frame_h, frame_w):
                    continue

                if self._est_arbitre(frame, bbox):
                    arbitres.append(Arbitre(
                        id=-1,
                        bbox=bbox,
                        position_terrain=pos,
                        confiance_detection=conf
                    ))
                else:
                    joueurs.append(Joueur(
                        id=-1,
                        equipe_id=-1,
                        bbox=bbox,
                        position_terrain=pos,
                        confiance_detection=conf,
                        role=RoleDetecte.JOUEUR
                    ))

        # Mise à jour des stats
        n = self._stats['frames'] + 1
        self._stats['frames']   = n
        self._stats['joueurs']  += len(joueurs)
        self._stats['ballons']  += 1 if ballon else 0
        self._stats['arbitres'] += len(arbitres)
        if confs_frame:
            old = self._stats['conf_moy']
            self._stats['conf_moy'] = (old * (n - 1) + np.mean(confs_frame)) / n

        return {
            'joueurs' : joueurs,
            'arbitres': arbitres,
            'ballon'  : ballon,
        }

    # ─────────────────────────────────────
    # FILTRES DE VALIDATION
    # ─────────────────────────────────────

    def _est_joueur_valide(self,
                            bbox: BoundingBox,
                            frame_h: int,
                            frame_w: int) -> bool:
        """Filtre les détections aberrantes (tribunes, panneaux, etc.)."""
        h = bbox.hauteur
        w = bbox.largeur

        # Ratio taille / frame
        if h / frame_h < 0.03 or h / frame_h > 0.75:
            return False
        if w / frame_w > 0.15:
            return False

        # Rapport hauteur/largeur : un joueur est plus grand que large
        if w > 0 and h / w < 1.0:
            return False

        # Pas dans les tribunes (bord supérieur)
        if bbox.y1 < frame_h * 0.08:
            return False

        return True

    def _est_arbitre(self, frame: np.ndarray, bbox: BoundingBox) -> bool:
        """
        Détecte si une personne est un arbitre
        via la couleur de maillot (jaune fluo typique).
        """
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        h = y2 - y1
        cy1 = y1 + int(h * 0.20)
        cy2 = y1 + int(h * 0.55)

        if cy1 >= cy2 or x1 >= x2:
            return False

        crop = frame[cy1:cy2, x1:x2]
        if crop.size == 0:
            return False

        hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        masque = cv2.inRange(hsv, ARBITRE_HSV_JAUNE_BAS, ARBITRE_HSV_JAUNE_HT)

        # Plus de 35 % de pixels jaunes dans la zone torse → arbitre
        return (np.sum(masque > 0) / masque.size) > 0.35

    def obtenir_stats(self) -> dict:
        return {**self._stats}

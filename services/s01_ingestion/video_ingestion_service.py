# services/s01_ingestion/video_ingestion_service.py

import cv2
import os
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Generator, Tuple
from core.base_service import BaseService

@dataclass
class VideoMetadata:
    chemin: str
    nom_fichier: str
    fps: float
    total_frames: int
    largeur: int
    hauteur: int
    duree_secondes: float
    duree_minutes: float
    taille_mo: float
    hash_md5: str
    codec: str
    resolutions_standard: str

@dataclass
class IngestionConfig:
    skip_frames: int = 3          # Analyser 1 frame sur N
    redimensionner: bool = True
    largeur_cible: int = 1280
    hauteur_cible: int = 720
    stabiliser: bool = False
    ameliorer_contraste: bool = True

class VideoIngestionService(BaseService):

    def __init__(self):
        super().__init__("VideoIngestion")
        self.metadata = None
        self.config = IngestionConfig()
        self.cap = None

    def initialiser(self):
        self.logger.info("Service Ingestion prêt")
        self.est_initialise = True

    def traiter(self, chemin_video: str) -> VideoMetadata:
        """Charger et analyser la vidéo"""
        return self.charger_video(chemin_video)

    # ─────────────────────────────────────
    # CHARGEMENT VIDÉO
    # ─────────────────────────────────────
    def charger_video(self, chemin: str) -> VideoMetadata:
        """Charger la vidéo et extraire ses métadonnées"""

        if not os.path.exists(chemin):
            raise FileNotFoundError(f"Vidéo introuvable: {chemin}")

        self.cap = cv2.VideoCapture(chemin)

        if not self.cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir: {chemin}")

        # Extraire métadonnées
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        largeur = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hauteur = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((codec_int >> 8*i) & 0xFF)
                        for i in range(4)])

        duree = total_frames / fps if fps > 0 else 0
        taille = os.path.getsize(chemin) / (1024 * 1024)

        # Identifier résolution
        if largeur >= 3840:
            res = "4K"
        elif largeur >= 1920:
            res = "Full HD"
        elif largeur >= 1280:
            res = "HD"
        else:
            res = "SD"

        # Hash MD5 pour identifier la vidéo
        hash_md5 = self._calculer_hash(chemin)

        self.metadata = VideoMetadata(
            chemin=chemin,
            nom_fichier=os.path.basename(chemin),
            fps=fps,
            total_frames=total_frames,
            largeur=largeur,
            hauteur=hauteur,
            duree_secondes=duree,
            duree_minutes=duree / 60,
            taille_mo=round(taille, 2),
            hash_md5=hash_md5,
            codec=codec,
            resolutions_standard=res
        )

        self.logger.info(f"📹 Vidéo: {self.metadata.nom_fichier}")
        self.logger.info(f"   Résolution: {largeur}x{hauteur} ({res})")
        self.logger.info(f"   FPS: {fps} | Durée: {duree/60:.1f} min")
        self.logger.info(f"   Frames: {total_frames} | Taille: {taille:.1f} Mo")

        return self.metadata

    # ─────────────────────────────────────
    # GÉNÉRATEUR DE FRAMES
    # ─────────────────────────────────────
    def generer_frames(self) -> Generator:
        """
        Générateur qui yield les frames une par une
        avec prétraitement optionnel
        """
        if self.cap is None:
            raise RuntimeError("Vidéo non chargée !")

        # Remettre au début
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_id = 0
        frames_traitees = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                break

            # Sauter des frames pour accélérer
            if frame_id % self.config.skip_frames == 0:
                timestamp = frame_id / self.metadata.fps

                # Prétraitement
                frame_traitee = self._pretraiter_frame(frame)

                yield frame_traitee, frame_id, timestamp
                frames_traitees += 1

            frame_id += 1

        self.logger.info(
            f"✅ {frames_traitees} frames traitées / {frame_id} total"
        )

    # ─────────────────────────────────────
    # PRÉTRAITEMENT FRAME
    # ─────────────────────────────────────
    def _pretraiter_frame(self, frame: np.ndarray) -> np.ndarray:
        """Améliorer la qualité de la frame"""

        # Redimensionner si nécessaire
        if self.config.redimensionner:
            if (frame.shape[1] != self.config.largeur_cible or
                frame.shape[0] != self.config.hauteur_cible):
                frame = cv2.resize(
                    frame,
                    (self.config.largeur_cible,
                     self.config.hauteur_cible)
                )

        # Améliorer le contraste
        if self.config.ameliorer_contraste:
            frame = self._ameliorer_contraste(frame)

        return frame

    def _ameliorer_contraste(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE - Amélioration adaptative du contraste"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    def _calculer_hash(self, chemin: str) -> str:
        """Calculer le hash MD5 de la vidéo"""
        hasher = hashlib.md5()
        with open(chemin, 'rb') as f:
            # Lire seulement les premiers Mo
            hasher.update(f.read(10 * 1024 * 1024))
        return hasher.hexdigest()

    def extraire_frame(self, frame_id: int) -> np.ndarray:
        """Extraire une frame spécifique"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        return frame if ret else None

    def liberer(self):
        """Libérer les ressources"""
        if self.cap:
            self.cap.release()
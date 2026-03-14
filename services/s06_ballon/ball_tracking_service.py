# services/s06_ballon/ball_tracking_service.py

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from collections import deque
from core.base_service import BaseService
from core.data_models import Ballon, Position, Joueur

class BallTrackingService(BaseService):
    """
    Tracker le ballon avec prédiction
    de sa position quand il n'est pas visible
    Calcule: vitesse, trajectoire, possession
    """

    POSSESSION_RADIUS_M = 3.0

    def __init__(self):
        super().__init__("BallTracking")
        self.fps = 25
        self.pixels_par_metre = 10.0

        # Historique des positions du ballon
        self.historique = deque(maxlen=500)
        self.derniere_position: Optional[Position] = None
        self.frames_perdues = 0
        self.MAX_FRAMES_PERDUES = 30

        # Vecteur de vitesse pour prédiction
        self.vecteur_vitesse = np.array([0.0, 0.0])

        # Stats ballon
        self.possession_frames: Dict[int, int] = {}  # equipe_id → nb frames
        self.vitesse_max = 0.0
        self.trajectoire_complete: List[Position] = []

    def initialiser(self):
        self.logger.info("✅ BallTracking initialisé")
        self.est_initialise = True

    def traiter(self, data: dict) -> dict:
        return self.mettre_a_jour(
            data.get('ballon'),
            data.get('joueurs', []),
            data['timestamp'],
            data['frame_id']
        )

    # ─────────────────────────────────────
    # MISE À JOUR
    # ─────────────────────────────────────
    def mettre_a_jour(self,
                       ballon: Optional[Ballon],
                       joueurs: List[Joueur],
                       timestamp: float,
                       frame_id: int) -> dict:
        """
        Mettre à jour la position du ballon
        """
        if ballon is not None:
            # Ballon détecté
            self.frames_perdues = 0

            # Mettre à jour vecteur vitesse
            if self.derniere_position is not None:
                dx = ballon.position.x - self.derniere_position.x
                dy = ballon.position.y - self.derniere_position.y
                self.vecteur_vitesse = np.array([dx, dy])

            self.derniere_position = ballon.position
            self.historique.append(ballon.position)
            self.trajectoire_complete.append(ballon.position)

            # Calculer vitesse
            vitesse = self._calculer_vitesse_ballon()
            ballon.vitesse = vitesse
            self.vitesse_max = max(self.vitesse_max, vitesse)

            # Déterminer possession
            possesseur = self._determiner_possession(
                ballon, joueurs
            )
            if possesseur:
                ballon.possesseur_id = possesseur.id
                ballon.equipe_possesseur = possesseur.equipe_id
                eq_id = possesseur.equipe_id
                self.possession_frames[eq_id] = (
                    self.possession_frames.get(eq_id, 0) + 1
                )

        else:
            # Ballon perdu - Prédire position
            self.frames_perdues += 1

            if (self.frames_perdues < self.MAX_FRAMES_PERDUES and
                    self.derniere_position is not None):
                # Prédiction par inertie
                position_predite = self._predire_position(timestamp, frame_id)
                ballon_predit = Ballon(
                    position=position_predite,
                    bbox=None,
                    confiance=0.0,
                    vitesse=0.0
                )
                return {
                    'ballon': ballon_predit,
                    'position_predite': True,
                    'frames_perdues': self.frames_perdues
                }

        return {
            'ballon': ballon,
            'position_predite': False,
            'frames_perdues': self.frames_perdues,
            'vitesse_actuelle': ballon.vitesse if ballon else 0
        }

    # ─────────────────────────────────────
    # PRÉDICTION DE POSITION
    # ─────────────────────────────────────
    def _predire_position(self,
                           timestamp: float,
                           frame_id: int) -> Position:
        """
        Prédire la position du ballon par inertie
        avec décélération progressive
        """
        facteur_deceleration = 0.85 ** self.frames_perdues
        vecteur_amorti = self.vecteur_vitesse * facteur_deceleration

        x_pred = self.derniere_position.x + vecteur_amorti[0]
        y_pred = self.derniere_position.y + vecteur_amorti[1]

        return Position(
            x=float(x_pred),
            y=float(y_pred),
            timestamp=timestamp,
            frame_id=frame_id
        )

    # ─────────────────────────────────────
    # POSSESSION DU BALLON
    # ─────────────────────────────────────
    def _determiner_possession(self,
                                ballon: Ballon,
                                joueurs: List[Joueur]) -> Optional[Joueur]:
        """
        Déterminer quel joueur possède le ballon
        basé sur la proximité
        """
        bx, by = ballon.position.x, ballon.position.y
        joueur_proche = None
        dist_min = float('inf')
        rayon_pixels = self.POSSESSION_RADIUS_M * max(self.pixels_par_metre, 1)

        for joueur in joueurs:
            jx = joueur.position_terrain.x
            jy = joueur.position_terrain.y
            dist = np.sqrt((bx - jx)**2 + (by - jy)**2)

            if dist < dist_min and dist < rayon_pixels:
                dist_min = dist
                joueur_proche = joueur

        return joueur_proche

    # ─────────────────────────────────────
    # STATISTIQUES BALLON
    # ─────────────────────────────────────
    def _calculer_vitesse_ballon(self) -> float:
        """Calculer vitesse du ballon en km/h"""
        if len(self.historique) < 2:
            return 0.0

        p1 = self.historique[-2]
        p2 = self.historique[-1]

        delta_t = p2.timestamp - p1.timestamp
        if delta_t <= 0:
            return 0.0

        dist_pixels = np.sqrt(
            (p2.x - p1.x)**2 + (p2.y - p1.y)**2
        )
        dist_metres = dist_pixels / self.pixels_par_metre
        vitesse_ms = dist_metres / delta_t
        vitesse_kmh = vitesse_ms * 3.6

        # Limiter les valeurs aberrantes (max ~120 km/h)
        return min(vitesse_kmh, 120.0)

    def get_stats_possession(self) -> dict:
        """Calculer les stats de possession"""
        total = sum(self.possession_frames.values())
        if total == 0:
            return {0: 50.0, 1: 50.0}

        return {
            eq_id: round((frames / total) * 100, 1)
            for eq_id, frames in self.possession_frames.items()
        }

    def get_trajectoire_recente(self, nb_points: int = 20) -> List:
        """Retourner les N dernières positions"""
        return list(self.historique)[-nb_points:]

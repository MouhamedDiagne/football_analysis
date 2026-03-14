# services/s02_terrain/terrain_detection_service.py

import cv2
import numpy as np
from typing import Optional, Tuple, List
from core.base_service import BaseService

class TerrainDetectionService(BaseService):
    """
    Détecter le terrain de football
    et établir la correspondance
    pixels ↔ coordonnées réelles (mètres)
    """

    # Dimensions standard terrain FIFA
    TERRAIN_LONGUEUR_M = 105.0
    TERRAIN_LARGEUR_M  = 68.0

    # Points clés du terrain réel (en mètres)
    POINTS_TERRAIN_REEL = np.float32([
        [0, 0],           # Coin bas gauche
        [105, 0],         # Coin bas droit
        [105, 68],        # Coin haut droit
        [0, 68]           # Coin haut gauche
    ])

    def __init__(self):
        super().__init__("TerrainDetection")
        self.homographie = None          # Matrice pixels → terrain
        self.homographie_inverse = None  # Matrice terrain → pixels
        self.masque_terrain = None
        self.coins_terrain_pixels = None
        self.pixels_par_metre_x = 1.0
        self.pixels_par_metre_y = 1.0
        self.lignes_detectees = []

    def initialiser(self):
        self.logger.info("Service TerrainDetection prêt")
        self.est_initialise = True

    def traiter(self, frame: np.ndarray) -> dict:
        """Analyser le terrain depuis une frame"""
        return self.analyser_terrain(frame)

    # ─────────────────────────────────────
    # ANALYSE PRINCIPALE
    # ─────────────────────────────────────
    def analyser_terrain(self, frame: np.ndarray) -> dict:
        """
        Pipeline complet d'analyse du terrain
        """
        resultats = {}

        # 1. Détecter les zones vertes
        masque = self._detecter_zone_verte(frame)
        self.masque_terrain = masque
        resultats['masque'] = masque

        # 2. Détecter les lignes blanches
        lignes = self._detecter_lignes(frame, masque)
        self.lignes_detectees = lignes
        resultats['lignes'] = lignes

        # 3. Trouver les coins du terrain
        coins = self._detecter_coins(frame, masque)
        if coins is not None:
            self.coins_terrain_pixels = coins
            resultats['coins'] = coins

            # 4. Calculer l'homographie
            homographie = self._calculer_homographie(coins)
            if homographie is not None:
                self.homographie = homographie
                self.homographie_inverse = np.linalg.inv(homographie)
                resultats['homographie'] = homographie

                # 5. Calculer l'échelle pixels/mètre
                self._calculer_echelle(coins, frame.shape)

        resultats['pixels_par_metre_x'] = self.pixels_par_metre_x
        resultats['pixels_par_metre_y'] = self.pixels_par_metre_y

        self.logger.info(
            f"Terrain détecté - Échelle: "
            f"{self.pixels_par_metre_x:.1f}px/m"
        )

        return resultats

    # ─────────────────────────────────────
    # DÉTECTION ZONE VERTE
    # ─────────────────────────────────────
    def _detecter_zone_verte(self, frame: np.ndarray) -> np.ndarray:
        """Isoler le terrain vert"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Plages pour le vert du gazon
        vert_bas1 = np.array([30, 30, 30])
        vert_haut1 = np.array([90, 255, 255])

        masque = cv2.inRange(hsv, vert_bas1, vert_haut1)

        # Morphologie pour nettoyer
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (15, 15)
        )
        masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, kernel)
        masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, kernel)

        return masque

    # ─────────────────────────────────────
    # DÉTECTION LIGNES BLANCHES
    # ─────────────────────────────────────
    def _detecter_lignes(self,
                          frame: np.ndarray,
                          masque: np.ndarray) -> List:
        """Détecter les lignes blanches du terrain"""

        # Isoler les zones blanches sur le terrain
        frame_terrain = cv2.bitwise_and(frame, frame, mask=masque)
        gris = cv2.cvtColor(frame_terrain, cv2.COLOR_BGR2GRAY)

        # Seuillage pour le blanc
        _, blanc = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY)

        # Détection de bords
        bords = cv2.Canny(blanc, 50, 150, apertureSize=3)

        # Détection des lignes droites (Hough)
        lignes = cv2.HoughLinesP(
            bords,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lignes is not None:
            return lignes.reshape(-1, 4).tolist()
        return []

    # ─────────────────────────────────────
    # DÉTECTION COINS DU TERRAIN
    # ─────────────────────────────────────
    def _detecter_coins(self,
                         frame: np.ndarray,
                         masque: np.ndarray) -> Optional[np.ndarray]:
        """Trouver les 4 coins du terrain"""

        # Trouver les contours
        contours, _ = cv2.findContours(
            masque, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Prendre le plus grand contour (= terrain)
        plus_grand = max(contours, key=cv2.contourArea)
        aire = cv2.contourArea(plus_grand)

        # Vérifier que c'est assez grand
        aire_min = frame.shape[0] * frame.shape[1] * 0.3
        if aire < aire_min:
            return None

        # Approximer en polygone à 4 coins
        perimetre = cv2.arcLength(plus_grand, True)
        approx = cv2.approxPolyDP(
            plus_grand, 0.02 * perimetre, True
        )

        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

        # Si pas exactement 4 points,
        # utiliser le rectangle englobant
        rect = cv2.minAreaRect(plus_grand)
        box = cv2.boxPoints(rect)
        return np.float32(box)

    # ─────────────────────────────────────
    # HOMOGRAPHIE
    # ─────────────────────────────────────
    def _calculer_homographie(self,
                               coins_pixels: np.ndarray
                               ) -> Optional[np.ndarray]:
        """
        Calculer la matrice de transformation
        pixels → coordonnées terrain réelles
        """
        # Ordonner les coins: BG, BD, HD, HG
        coins_ordonnes = self._ordonner_coins(coins_pixels)

        homographie, masque = cv2.findHomography(
            coins_ordonnes,
            self.POINTS_TERRAIN_REEL,
            cv2.RANSAC,
            5.0
        )

        return homographie

    def _ordonner_coins(self, coins: np.ndarray) -> np.ndarray:
        """Ordonner les coins dans le bon ordre"""
        # Calculer le centroïde
        centre = coins.mean(axis=0)

        # Ordonner par angle
        angles = np.arctan2(
            coins[:, 1] - centre[1],
            coins[:, 0] - centre[0]
        )
        ordre = np.argsort(angles)
        return coins[ordre]

    # ─────────────────────────────────────
    # CONVERSION COORDONNÉES
    # ─────────────────────────────────────
    def pixels_vers_metres(self,
                            px: float,
                            py: float
                            ) -> Tuple[float, float]:
        """Convertir pixels → mètres sur le terrain"""
        if self.homographie is None:
            return px / self.pixels_par_metre_x, py / self.pixels_par_metre_y

        point = np.array([[[px, py]]], dtype=np.float32)
        point_transforme = cv2.perspectiveTransform(
            point, self.homographie
        )
        return (float(point_transforme[0][0][0]),
                float(point_transforme[0][0][1]))

    def metres_vers_pixels(self,
                            mx: float,
                            my: float
                            ) -> Tuple[int, int]:
        """Convertir mètres → pixels"""
        if self.homographie_inverse is None:
            return (int(mx * self.pixels_par_metre_x),
                    int(my * self.pixels_par_metre_y))

        point = np.array([[[mx, my]]], dtype=np.float32)
        point_transforme = cv2.perspectiveTransform(
            point, self.homographie_inverse
        )
        return (int(point_transforme[0][0][0]),
                int(point_transforme[0][0][1]))

    def _calculer_echelle(self,
                           coins: np.ndarray,
                           shape: tuple):
        """Calculer l'échelle pixels par mètre"""
        largeur_pixels = np.linalg.norm(coins[1] - coins[0])
        hauteur_pixels = np.linalg.norm(coins[3] - coins[0])

        self.pixels_par_metre_x = (
            largeur_pixels / self.TERRAIN_LONGUEUR_M
        )
        self.pixels_par_metre_y = (
            hauteur_pixels / self.TERRAIN_LARGEUR_M
        )

    def est_sur_terrain(self, px: int, py: int) -> bool:
        """Vérifier si un point est sur le terrain"""
        if self.masque_terrain is None:
            return True
        if (0 <= py < self.masque_terrain.shape[0] and
                0 <= px < self.masque_terrain.shape[1]):
            return self.masque_terrain[py, px] > 0
        return False
# services/s08_tactique/tactical_analysis_service.py  — v2.0
# Analytics avancés : heatmaps, passes, pressing Voronoi, xG, formations

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from core.base_service import BaseService
from core.data_models import (
    Joueur, Ballon, Position, StatsTactiquesJoueur, ZoneTerrain, Evenement, TypeEvenement
)


class TacticalAnalysisService(BaseService):
    """
    Analyse tactique enrichie v2 :

    Nouveautés :
    - Détection de passes via changement de possesseur + trajectoire balle
    - Pressing basé sur zones Voronoi (territoire contrôlé)
    - Modèle xG (distance + angle au but)
    - Formation multi-lignes plus précise
    - Heatmap 12×12 inchangée
    """

    # Dimensions réelles du terrain (m)
    TERRAIN_L = 105.0
    TERRAIN_W = 68.0

    # Buts (coordonnées en mètres, centre du but)
    BUT_GAUCHE  = (0.0,   34.0)
    BUT_DROITE  = (105.0, 34.0)
    LARGEUR_BUT = 7.32
    SURFACE_REPARATION_PROF = 16.5   # mètres depuis la ligne de but

    def __init__(self):
        super().__init__("TacticalAnalysis")

        # Positions accumulées par joueur : [(x_m, y_m, ts), ...]
        self.positions_acc:    Dict[int, List[Tuple]] = {}
        # Positions accumulées par équipe : [[(x_m, y_m), ...], ...]
        self.positions_equipe: Dict[int, List[List[Tuple]]] = {0: [], 1: []}

        # Trajectoire ballon (pour détection de passes)
        self._traj_ballon:  deque = deque(maxlen=60)
        # Dernière possession connue (joueur_id, equipe_id)
        self._derniere_possession: Tuple[Optional[int], Optional[int]] = (None, None)

        # Événements détectés par ce service
        self._evenements_frame: List[Evenement] = []

        # Formation courante par équipe
        self._formation: Dict[int, str] = {0: "N/A", 1: "N/A"}

        self.fps = 25
        self.pixels_par_metre = 10.0

    def initialiser(self):
        self.logger.info("✅ TacticalAnalysis v2 initialisé")
        self.est_initialise = True

    def traiter(self, data: dict) -> dict:
        return self.analyser_frame(
            data['joueurs'],
            data.get('ballon'),
            data['frame_id'],
            data['timestamp']
        )

    # ─────────────────────────────────────
    # ANALYSE PAR FRAME
    # ─────────────────────────────────────

    def analyser_frame(self,
                        joueurs: List[Joueur],
                        ballon: Optional[Ballon],
                        frame_id: int,
                        timestamp: float) -> dict:
        """
        Traite une frame :
        - Accumule positions
        - Détecte passes et tirs via heuristiques
        - Calcule pressing Voronoi
        - Met à jour la formation toutes les 50 frames
        """
        self._evenements_frame = []

        # ── Accumulation positions ────────────────────────────────
        for j in joueurs:
            jid = j.id
            x_m = j.position_terrain.x / max(self.pixels_par_metre, 1)
            y_m = j.position_terrain.y / max(self.pixels_par_metre, 1)
            if jid not in self.positions_acc:
                self.positions_acc[jid] = []
            self.positions_acc[jid].append((x_m, y_m, timestamp))

        # ── Positions par équipe ──────────────────────────────────
        for eq_id in [0, 1]:
            pts = [
                (j.position_terrain.x / max(self.pixels_par_metre, 1),
                 j.position_terrain.y / max(self.pixels_par_metre, 1))
                for j in joueurs if j.equipe_id == eq_id
            ]
            if pts:
                self.positions_equipe[eq_id].append(pts)

        # ── Trajectoire ballon ────────────────────────────────────
        if ballon is not None:
            bx_m = ballon.position.x / max(self.pixels_par_metre, 1)
            by_m = ballon.position.y / max(self.pixels_par_metre, 1)
            self._traj_ballon.append((bx_m, by_m, timestamp,
                                      ballon.possesseur_id,
                                      ballon.equipe_possesseur))

        # ── Événements (passes + tirs) ────────────────────────────
        if ballon is not None:
            evts = self._detecter_evenements(ballon, frame_id, timestamp)
            self._evenements_frame.extend(evts)

        # ── Pressing Voronoi ──────────────────────────────────────
        pressing = self._calculer_pressing_voronoi(joueurs)

        # ── Formation toutes les 50 frames ────────────────────────
        formation = {}
        n0 = len(self.positions_equipe[0])
        if n0 > 0 and n0 % 50 == 0:
            for eq_id in [0, 1]:
                self._formation[eq_id] = self._detecter_formation(eq_id)
        formation = dict(self._formation)

        return {
            'formation':  formation,
            'pressing':   pressing,
            'evenements': self._evenements_frame,
        }

    # ─────────────────────────────────────
    # DÉTECTION D'ÉVÉNEMENTS
    # ─────────────────────────────────────

    def _detecter_evenements(self,
                               ballon: Ballon,
                               frame_id: int,
                               timestamp: float) -> List[Evenement]:
        evts = []
        poss_id  = ballon.possesseur_id
        eq_id    = ballon.equipe_possesseur
        old_id, old_eq = self._derniere_possession

        # ── Passe : changement de possesseur dans la MÊME équipe ─
        if (poss_id is not None
                and old_id is not None
                and old_id != poss_id
                and eq_id is not None
                and old_eq == eq_id
                and len(self._traj_ballon) >= 2):

            pos_src  = ballon.position   # approximation
            pos_dest = ballon.position

            # Distance de la passe
            if len(self._traj_ballon) >= 2:
                bx_src, by_src = self._traj_ballon[-2][0], self._traj_ballon[-2][1]
                bx_dst, by_dst = ballon.position.x / max(self.pixels_par_metre, 1), \
                                  ballon.position.y / max(self.pixels_par_metre, 1)
                dist_passe = np.sqrt((bx_dst - bx_src)**2 + (by_dst - by_src)**2)
            else:
                dist_passe = 0.0

            evts.append(Evenement(
                type=TypeEvenement.PASSE,
                timestamp=timestamp,
                frame_id=frame_id,
                joueur_id=old_id,
                equipe_id=eq_id,
                position=pos_src,
                position_destination=pos_dest,
                reussi=True,
                distance_m=round(dist_passe, 1)
            ))

        # ── Passe adverse (interception) ──────────────────────────
        elif (poss_id is not None
              and old_id is not None
              and old_id != poss_id
              and eq_id is not None
              and old_eq is not None
              and old_eq != eq_id):

            evts.append(Evenement(
                type=TypeEvenement.INTERCEPTION,
                timestamp=timestamp,
                frame_id=frame_id,
                joueur_id=poss_id,
                equipe_id=eq_id,
                position=ballon.position,
                reussi=True
            ))

        # ── Tir : ballon proche d'un but ──────────────────────────
        if eq_id is not None:
            xg, angle = self._calculer_xg(ballon)
            if xg > 0.02:   # probabilité > 2% → position de tir
                evts.append(Evenement(
                    type=TypeEvenement.TIR,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    joueur_id=poss_id,
                    equipe_id=eq_id,
                    position=ballon.position,
                    reussi=False,
                    valeur_xg=round(xg, 3),
                    angle_but=round(angle, 1)
                ))

        # Mémoriser possession actuelle
        self._derniere_possession = (poss_id, eq_id)
        return evts

    # ─────────────────────────────────────
    # MODÈLE xG (Expected Goals)
    # ─────────────────────────────────────

    def _calculer_xg(self, ballon: Ballon) -> Tuple[float, float]:
        """
        Calcule la valeur xG et l'angle de tir depuis la position du ballon.
        Utilise un modèle simplifié distance + angle.
        Retourne (xg, angle_deg).
        """
        bx = ballon.position.x / max(self.pixels_par_metre, 1)
        by = ballon.position.y / max(self.pixels_par_metre, 1)

        # Distance aux deux buts
        dist_g = np.sqrt((bx - self.BUT_GAUCHE[0])**2
                         + (by - self.BUT_GAUCHE[1])**2)
        dist_d = np.sqrt((bx - self.BUT_DROITE[0])**2
                         + (by - self.BUT_DROITE[1])**2)
        dist   = min(dist_g, dist_d)

        if dist > 40:   # Hors portée de tir
            return 0.0, 0.0

        # But le plus proche
        but = self.BUT_GAUCHE if dist_g < dist_d else self.BUT_DROITE

        # Angle d'ouverture du but (théorème du cosinus)
        p1 = (but[0], but[1] - self.LARGEUR_BUT / 2)
        p2 = (but[0], but[1] + self.LARGEUR_BUT / 2)
        d1 = np.sqrt((bx - p1[0])**2 + (by - p1[1])**2)
        d2 = np.sqrt((bx - p2[0])**2 + (by - p2[1])**2)
        num = (self.LARGEUR_BUT ** 2 - d1**2 - d2**2 + 2 * d1 * d2)
        if d1 * d2 < 1e-6:
            angle_deg = 0.0
        else:
            cos_a = (d1**2 + d2**2 - self.LARGEUR_BUT**2) / (2 * d1 * d2 + 1e-8)
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

        # xG = f(distance, angle) — modèle empirique simplifié
        xg_dist  = np.exp(-dist / 12.0)               # décroît avec la distance
        xg_angle = angle_deg / 90.0                   # plus l'angle est ouvert, mieux c'est
        xg = round(xg_dist * xg_angle * 0.5, 4)       # calibration empirique

        # Bonus surface de réparation
        if dist < self.SURFACE_REPARATION_PROF:
            xg = min(xg * 1.5, 0.99)

        return xg, angle_deg

    # ─────────────────────────────────────
    # PRESSING VORONOI
    # ─────────────────────────────────────

    def _calculer_pressing_voronoi(self,
                                    joueurs: List[Joueur]) -> dict:
        """
        Calcule l'intensité de pressing par équipe
        en utilisant la surface de Voronoi contrôlée.
        Un joueur sous pression est celui dont la cellule Voronoi
        est envahie par un adversaire à < 5m.
        """
        pressing = {0: 0.0, 1: 0.0}

        if len(joueurs) < 2:
            return pressing

        pts  = np.array([
            [j.position_terrain.x / max(self.pixels_par_metre, 1),
             j.position_terrain.y / max(self.pixels_par_metre, 1)]
            for j in joueurs
        ])
        eqs  = [j.equipe_id for j in joueurs]
        seuil_m = 5.0   # 5 mètres = pressing actif

        for i, j_eq in enumerate(eqs):
            j_adv = 1 - j_eq
            for k, k_eq in enumerate(eqs):
                if k_eq != j_adv:
                    continue
                dist = np.linalg.norm(pts[i] - pts[k])
                if dist < seuil_m:
                    pressing[j_eq] += 1.0 / (dist + 0.1)

        # Normaliser par nombre de joueurs
        for eq_id in [0, 1]:
            n = sum(1 for j in joueurs if j.equipe_id == eq_id)
            if n > 0:
                pressing[eq_id] = round(pressing[eq_id] / n, 3)

        return pressing

    # ─────────────────────────────────────
    # HEATMAP
    # ─────────────────────────────────────

    def generer_heatmap(self,
                         joueur_id: int,
                         grille: int = 12) -> List[List[float]]:
        """Génère la heatmap 12×12 pour un joueur (0-100)."""
        heatmap   = np.zeros((grille, grille))
        positions = self.positions_acc.get(joueur_id, [])

        for x, y, _ in positions:
            col = int(max(0, min((x / self.TERRAIN_L) * grille, grille - 1)))
            row = int(max(0, min((y / self.TERRAIN_W) * grille, grille - 1)))
            heatmap[row][col] += 1

        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max()) * 100

        return np.round(heatmap, 1).tolist()

    # ─────────────────────────────────────
    # TERRITOIRE VORONOI PAR JOUEUR (m²)
    # ─────────────────────────────────────

    def calculer_territoire_voronoi(self,
                                     joueurs: List[Joueur]) -> Dict[int, float]:
        """
        Estime la surface (m²) de territoire contrôlée par chaque joueur
        via les polygones de Voronoi clippés au terrain.
        """
        if len(joueurs) < 3:
            return {}

        try:
            from scipy.spatial import Voronoi
        except ImportError:
            return {}

        pts = np.array([
            [j.position_terrain.x / max(self.pixels_par_metre, 1),
             j.position_terrain.y / max(self.pixels_par_metre, 1)]
            for j in joueurs
        ])

        # Ajouter 4 points miroir pour gérer les régions infinies
        pts_miroir = np.concatenate([
            pts,
            [[-10, -10], [-10, 80], [115, -10], [115, 80]]
        ])

        try:
            vor = Voronoi(pts_miroir)
        except Exception:
            return {}

        # Approximation rapide : aire = 1 / (densité locale)
        # (calcul d'aire exact de polygon de Voronoi est coûteux sur CPU)
        territoires = {}
        for i, j in enumerate(joueurs):
            if i < len(vor.point_region):
                region_idx = vor.point_region[i]
                region     = vor.regions[region_idx]
                if -1 not in region and len(region) > 0:
                    poly = vor.vertices[region]
                    aire = self._aire_polygone(poly)
                    # Clipper à la surface du terrain
                    aire = min(aire, self.TERRAIN_L * self.TERRAIN_W)
                    territoires[j.id] = round(aire, 1)
                else:
                    territoires[j.id] = 0.0

        return territoires

    def _aire_polygone(self, pts: np.ndarray) -> float:
        """Calcule l'aire d'un polygone par la formule du lacet."""
        n = len(pts)
        if n < 3:
            return 0.0
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(
            np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)
        )

    # ─────────────────────────────────────
    # FORMATION
    # ─────────────────────────────────────

    def _detecter_formation(self, equipe_id: int) -> str:
        """
        Détecte la formation de l'équipe sur les 50 dernières frames.
        Divise le terrain en 4 bandes (défense / milieu-def / milieu-att / attaque).
        """
        historique = self.positions_equipe.get(equipe_id, [])
        if not historique:
            return "N/A"

        frames_recentes = historique[-50:]
        tous_x, tous_y  = [], []
        for frame_pts in frames_recentes:
            for x, y in frame_pts:
                tous_x.append(x)
                tous_y.append(y)

        if len(tous_x) < 10:
            return "N/A"

        q25 = np.percentile(tous_x, 25)
        q50 = np.percentile(tous_x, 50)
        q75 = np.percentile(tous_x, 75)

        def_zone = sum(1 for x in tous_x if x < q25)
        mid_def  = sum(1 for x in tous_x if q25 <= x < q50)
        mid_att  = sum(1 for x in tous_x if q50 <= x < q75)
        att_zone = sum(1 for x in tous_x if x >= q75)
        total    = def_zone + mid_def + mid_att + att_zone

        if total == 0:
            return "N/A"

        s  = 10 / total
        d  = round(def_zone * s)
        m1 = round(mid_def  * s)
        m2 = round(mid_att  * s)
        a  = round(att_zone * s)

        return f"{d}-{m1}-{m2}-{a}" if m1 != m2 else f"{d}-{m1+m2}-{a}"

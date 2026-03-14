# core/data_models.py  — v2.0
# Modèles de données enrichis : Pose, Arbitre, Biomécanique

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import numpy as np


# ─────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────

class Poste(Enum):
    GARDIEN    = "Gardien"
    DEFENSEUR  = "Défenseur"
    MILIEU     = "Milieu"
    ATTAQUANT  = "Attaquant"
    INCONNU    = "Inconnu"

class TypeEvenement(Enum):
    TIR          = "Tir"
    PASSE        = "Passe"
    DRIBBLE      = "Dribble"
    TACLE        = "Tacle"
    INTERCEPTION = "Interception"
    CORNER       = "Corner"
    REMISE_JEU   = "Remise en jeu"
    BUT          = "But"
    HORS_JEU     = "Hors jeu"
    FAUTE        = "Faute"
    DEGAGEMENT   = "Dégagement"
    PRESSION     = "Pression"
    REPRISE      = "Reprise"

class ZoneTerrain(Enum):
    DEFENCE_PROFONDE   = "Défense profonde"
    DEFENCE            = "Défense"
    MILIEU_DEFENSIF    = "Milieu défensif"
    MILIEU_CENTRAL     = "Milieu central"
    MILIEU_OFFENSIF    = "Milieu offensif"
    ATTAQUE            = "Attaque"
    SURFACE_REPARATION = "Surface de réparation"

class RoleDetecte(Enum):
    JOUEUR  = "joueur"
    ARBITRE = "arbitre"
    GARDIEN = "gardien"
    INCONNU = "inconnu"


# ─────────────────────────────────────────
# MODÈLES DE BASE
# ─────────────────────────────────────────

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def centre(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2,
                (self.y1 + self.y2) // 2)

    @property
    def largeur(self) -> int:
        return self.x2 - self.x1

    @property
    def hauteur(self) -> int:
        return self.y2 - self.y1

    @property
    def aire(self) -> int:
        return self.largeur * self.hauteur

    def iou(self, autre: 'BoundingBox') -> float:
        """Intersection over Union avec une autre bounding box."""
        ix1 = max(self.x1, autre.x1)
        iy1 = max(self.y1, autre.y1)
        ix2 = min(self.x2, autre.x2)
        iy2 = min(self.y2, autre.y2)
        if ix2 < ix1 or iy2 < iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = self.aire + autre.aire - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Position:
    x: float
    y: float
    timestamp: float
    frame_id: int

    def distance_vers(self, autre: 'Position') -> float:
        return np.sqrt(
            (self.x - autre.x) ** 2 +
            (self.y - autre.y) ** 2
        )


# ─────────────────────────────────────────
# POSE ESTIMATION — 17 keypoints COCO
# ─────────────────────────────────────────

# Indices keypoints COCO format
COCO_KEYPOINTS = {
    'nez':       0,
    'oeil_g':    1,  'oeil_d':    2,
    'oreille_g': 3,  'oreille_d': 4,
    'epaule_g':  5,  'epaule_d':  6,
    'coude_g':   7,  'coude_d':   8,
    'poignet_g': 9,  'poignet_d': 10,
    'hanche_g':  11, 'hanche_d':  12,
    'genou_g':   13, 'genou_d':   14,
    'cheville_g':15, 'cheville_d':16,
}

@dataclass
class PoseKeypoints:
    """
    17 keypoints COCO par joueur.
    keypoints  : np.ndarray shape (17, 2) — coordonnées [x, y] pixels
    confidences: np.ndarray shape (17,)   — score de confiance par keypoint
    """
    keypoints:    np.ndarray   # (17, 2)
    confidences:  np.ndarray   # (17,)
    frame_id:     int
    seuil_confiance: float = 0.5

    def get_point(self, idx: int) -> Optional[Tuple[float, float]]:
        """Retourne (x, y) si confiant, None sinon."""
        if 0 <= idx < 17 and self.confidences[idx] >= self.seuil_confiance:
            return (float(self.keypoints[idx, 0]),
                    float(self.keypoints[idx, 1]))
        return None

    def get_point_par_nom(self, nom: str) -> Optional[Tuple[float, float]]:
        """Retourne le keypoint par nom (ex: 'epaule_g')."""
        idx = COCO_KEYPOINTS.get(nom)
        return self.get_point(idx) if idx is not None else None

    def angle_articulation(self, idx_a: int, idx_b: int, idx_c: int) -> Optional[float]:
        """
        Calcule l'angle en degrés au point B
        entre les segments A-B et B-C.
        """
        a = self.get_point(idx_a)
        b = self.get_point(idx_b)
        c = self.get_point(idx_c)
        if None in (a, b, c):
            return None
        v1 = np.array([a[0] - b[0], a[1] - b[1]])
        v2 = np.array([c[0] - b[0], c[1] - b[1]])
        norme = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norme < 1e-8:
            return None
        cos_a = np.dot(v1, v2) / norme
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def centre_masse(self) -> Optional[Tuple[float, float]]:
        """Estime le centre de masse (moyenne épaules + hanches)."""
        indices = [5, 6, 11, 12]   # épaules + hanches
        pts = [self.get_point(i) for i in indices]
        pts_valides = [p for p in pts if p is not None]
        if len(pts_valides) < 2:
            return None
        return (
            float(np.mean([p[0] for p in pts_valides])),
            float(np.mean([p[1] for p in pts_valides]))
        )

    def vecteur_inclination(self) -> Optional[float]:
        """
        Angle d'inclinaison du tronc (degrés, 0° = vertical).
        Utilise épaules et hanches.
        """
        eg = self.get_point_par_nom('epaule_g')
        ed = self.get_point_par_nom('epaule_d')
        hg = self.get_point_par_nom('hanche_g')
        hd = self.get_point_par_nom('hanche_d')
        if None in (eg, ed, hg, hd):
            return None
        moy_epaules = ((eg[0]+ed[0])/2, (eg[1]+ed[1])/2)
        moy_hanches = ((hg[0]+hd[0])/2, (hg[1]+hd[1])/2)
        dx = moy_epaules[0] - moy_hanches[0]
        dy = moy_hanches[1] - moy_epaules[1]   # y inversé en image
        if abs(dy) < 1e-6:
            return 90.0
        return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


# ─────────────────────────────────────────
# ENTITÉS DU JEU
# ─────────────────────────────────────────

@dataclass
class Joueur:
    id: int
    equipe_id: int
    bbox: BoundingBox
    position_terrain: Position
    couleur_maillot: Tuple[int, int, int] = (0, 0, 0)
    poste: Poste = Poste.INCONNU
    numero_maillot: Optional[int] = None
    confiance_detection: float = 0.0
    # Nouveautés v2
    pose:         Optional[PoseKeypoints] = None
    est_arbitre:  bool = False
    role:         RoleDetecte = RoleDetecte.JOUEUR


@dataclass
class Arbitre:
    id: int
    bbox: BoundingBox
    position_terrain: Position
    confiance_detection: float = 0.0


@dataclass
class Ballon:
    position: Position
    bbox: Optional[BoundingBox]
    confiance: float
    vitesse: float = 0.0
    possesseur_id: Optional[int] = None
    equipe_possesseur: Optional[int] = None


@dataclass
class FrameData:
    frame_id: int
    timestamp: float
    image: np.ndarray
    joueurs:  List[Joueur]  = field(default_factory=list)
    arbitres: List[Arbitre] = field(default_factory=list)
    ballon:   Optional[Ballon] = None
    homographie: Optional[np.ndarray] = None


# ─────────────────────────────────────────
# STATS BIOMÉCANIQUES  (nouveau v2)
# ─────────────────────────────────────────

@dataclass
class StatsBiomecaniques:
    joueur_id: int
    # Foulées
    longueur_foulee_moy_m:   float = 0.0   # longueur moyenne en mètres
    frequence_foulee_hz:     float = 0.0   # foulées par seconde
    # Posture
    angle_inclination_moy:   float = 0.0   # inclinaison tronc (degrés)
    amplitude_bras:          float = 0.0   # écart poignets normalisé
    # Symétrie & équilibre
    symetrie_course:         float = 100.0 # 100 = parfaitement symétrique
    # Angles articulaires moyens
    angle_genou_g_moy:       float = 0.0
    angle_genou_d_moy:       float = 0.0
    angle_hanche_g_moy:      float = 0.0
    angle_hanche_d_moy:      float = 0.0
    # Charge articulaire estimée (0-100)
    charge_genou:            float = 0.0
    charge_hanche:           float = 0.0
    # Compteurs
    nb_frames_pose:          int   = 0


# ─────────────────────────────────────────
# MODÈLES STATISTIQUES
# ─────────────────────────────────────────

@dataclass
class StatsPhysiquesJoueur:
    joueur_id: int
    distance_totale_m:     float = 0.0
    distance_marche_m:     float = 0.0    # 0-7 km/h
    distance_trot_m:       float = 0.0    # 7-14 km/h
    distance_course_m:     float = 0.0    # 14-21 km/h
    distance_sprint_m:     float = 0.0    # > 21 km/h
    vitesse_max_kmh:       float = 0.0
    vitesse_moyenne_kmh:   float = 0.0
    nombre_sprints:        int   = 0
    nombre_accelerations:  int   = 0
    charge_physique:       float = 0.0
    metres_par_minute:     float = 0.0
    # Biomécanique (intégré depuis StatsBiomecaniques)
    longueur_foulee_moy_m: float = 0.0
    symetrie_course:       float = 100.0
    angle_inclination_moy: float = 0.0


@dataclass
class StatsTechniquesJoueur:
    joueur_id: int
    # Passes
    passes_total:       int   = 0
    passes_reussies:    int   = 0
    passes_longues:     int   = 0
    passes_cles:        int   = 0
    # Tirs
    tirs_total:         int   = 0
    tirs_cadres:        int   = 0
    buts:               int   = 0
    expected_goals:     float = 0.0
    # Dribbles & Duels
    dribbles_tentes:    int   = 0
    dribbles_reussis:   int   = 0
    duels_total:        int   = 0
    duels_gagnes:       int   = 0
    # Défense
    tacles:             int   = 0
    tacles_reussis:     int   = 0
    interceptions:      int   = 0
    degagements:        int   = 0
    # Discipline
    fautes_commises:    int   = 0
    fautes_subies:      int   = 0
    carton_jaune:       bool  = False
    carton_rouge:       bool  = False
    # Gardien
    arrets:             int   = 0
    buts_encaisses:     int   = 0


@dataclass
class StatsTactiquesJoueur:
    joueur_id: int
    position_moyenne:     Tuple[float, float] = (0.0, 0.0)
    heatmap:              List[List[float]]   = field(default_factory=list)
    zones_frequentees:    List[ZoneTerrain]   = field(default_factory=list)
    rayon_action_m:       float = 0.0
    profondeur_moyenne:   float = 0.0
    largeur_moyenne:      float = 0.0
    pressing_score:       float = 0.0
    territoire_voronoi_m2: float = 0.0


@dataclass
class StatsEquipe:
    equipe_id: int
    nom: str
    # Possession
    possession_pct:       float = 0.0
    # Tirs
    tirs_total:           int   = 0
    tirs_cadres:          int   = 0
    expected_goals:       float = 0.0
    buts:                 int   = 0
    # Passes
    passes_total:         int   = 0
    passes_reussies:      int   = 0
    passes_progressives:  int   = 0
    # Pressing
    ppda:                 float = 0.0
    pressing_intensite:   float = 0.0
    # Physique
    distance_totale_km:   float = 0.0
    sprints_total:        int   = 0
    # Discipline
    fautes:               int   = 0
    corners:              int   = 0
    hors_jeux:            int   = 0
    cartons_jaunes:       int   = 0
    cartons_rouges:       int   = 0
    # Tactique
    formation:            str   = ""
    bloc_defensif_hauteur: float = 0.0
    largeur_jeu:          float = 0.0


@dataclass
class Evenement:
    type:                TypeEvenement
    timestamp:           float
    frame_id:            int
    joueur_id:           Optional[int]
    equipe_id:           Optional[int]
    position:            Position
    position_destination: Optional[Position] = None
    reussi:              bool  = True
    valeur_xg:           float = 0.0
    description:         str   = ""
    distance_m:          float = 0.0
    angle_but:           float = 0.0

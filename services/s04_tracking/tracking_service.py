# services/s04_tracking/tracking_service.py  — v2.0
# Tracking multi-objets avec ByteTrack (supervision)

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from core.base_service import BaseService
from core.data_models import Joueur, Arbitre, BoundingBox, Position


class TrackingService(BaseService):
    """
    Suivi multi-objets avec ByteTrack (via supervision).
    Maintient un ID unique par joueur/arbitre sur toute la séquence.

    Avantages sur DeepSORT :
    - Pas de ré-identification par apparence : plus rapide sur CPU
    - Meilleure gestion des occlusions via double-buffer à faible score
    - Interface unifiée supervision
    """

    def __init__(self, fps: int = 25):
        super().__init__("Tracking")
        self.fps              = fps
        self.tracker_joueurs  = None
        self.tracker_arbitres = None

        # { track_id : [Position, ...] }
        self.historique_positions: Dict[int, List[Position]] = defaultdict(list)
        self.historique_arbitres:  Dict[int, List[Position]] = defaultdict(list)

        # track_id → equipe_id
        self.equipe_par_id: Dict[int, int] = {}

        self.nb_joueurs_uniques = 0

    def initialiser(self):
        import supervision as sv

        self.tracker_joueurs = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=self.fps
        )
        self.tracker_arbitres = sv.ByteTrack(
            track_activation_threshold=0.30,
            lost_track_buffer=20,
            minimum_matching_threshold=0.8,
            frame_rate=self.fps
        )
        self.logger.info("✅ ByteTrack initialisé (supervision)")
        self.est_initialise = True

    def traiter(self, data: dict) -> dict:
        joueurs_suivis = self.mettre_a_jour(
            data['frame'],
            data.get('joueurs', []),
            data['timestamp'],
            data['frame_id']
        )
        arbitres_suivis = self.mettre_a_jour_arbitres(
            data.get('arbitres', []),
            data['timestamp'],
            data['frame_id']
        )
        return {'joueurs': joueurs_suivis, 'arbitres': arbitres_suivis}

    # ─────────────────────────────────────
    # TRACKING JOUEURS
    # ─────────────────────────────────────

    def mettre_a_jour(self,
                       frame: np.ndarray,
                       joueurs: List[Joueur],
                       timestamp: float,
                       frame_id: int) -> List[Joueur]:
        """Met à jour ByteTrack pour les joueurs. Retourne la liste trackée."""
        import supervision as sv

        if not joueurs:
            return []

        xyxy  = np.array(
            [[j.bbox.x1, j.bbox.y1, j.bbox.x2, j.bbox.y2] for j in joueurs],
            dtype=np.float32
        )
        confs = np.array([j.confiance_detection for j in joueurs], dtype=np.float32)
        clses = np.zeros(len(joueurs), dtype=int)

        detections = sv.Detections(xyxy=xyxy, confidence=confs, class_id=clses)
        tracked    = self.tracker_joueurs.update_with_detections(detections)

        return self._construire_joueurs(
            tracked, joueurs, timestamp, frame_id, self.historique_positions
        )

    # ─────────────────────────────────────
    # TRACKING ARBITRES
    # ─────────────────────────────────────

    def mettre_a_jour_arbitres(self,
                                arbitres: List[Arbitre],
                                timestamp: float,
                                frame_id: int) -> List[Arbitre]:
        """Met à jour ByteTrack pour les arbitres."""
        import supervision as sv

        if not arbitres:
            return []

        xyxy  = np.array(
            [[a.bbox.x1, a.bbox.y1, a.bbox.x2, a.bbox.y2] for a in arbitres],
            dtype=np.float32
        )
        confs = np.array([a.confiance_detection for a in arbitres], dtype=np.float32)
        clses = np.zeros(len(arbitres), dtype=int)

        detections = sv.Detections(xyxy=xyxy, confidence=confs, class_id=clses)
        tracked    = self.tracker_arbitres.update_with_detections(detections)

        arbitres_suivis = []
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pos    = Position(cx, cy, timestamp, frame_id)
            self.historique_arbitres[tid].append(pos)

            src  = self._trouver_source(arbitres, cx, cy)
            conf = src.confiance_detection if src else 0.5
            arbitres_suivis.append(Arbitre(
                id=tid,
                bbox=BoundingBox(x1, y1, x2, y2),
                position_terrain=pos,
                confiance_detection=conf
            ))

        return arbitres_suivis

    # ─────────────────────────────────────
    # UTILITAIRES INTERNES
    # ─────────────────────────────────────

    def _construire_joueurs(self,
                             tracked,
                             joueurs_bruts: List[Joueur],
                             timestamp: float,
                             frame_id: int,
                             historique: Dict) -> List[Joueur]:
        """Construit la liste de joueurs trackés depuis le résultat ByteTrack."""
        joueurs_suivis = []

        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pos    = Position(cx, cy, timestamp, frame_id)
            historique[tid].append(pos)

            src = self._trouver_source(joueurs_bruts, cx, cy)
            if src:
                src.id             = tid
                src.position_terrain = pos
                src.bbox           = BoundingBox(x1, y1, x2, y2)
                # Restaurer l'équipe précédemment connue si pas encore classifiée
                if src.equipe_id < 0 and tid in self.equipe_par_id:
                    src.equipe_id = self.equipe_par_id[tid]
                joueur_final = src
            else:
                joueur_final = Joueur(
                    id=tid,
                    equipe_id=self.equipe_par_id.get(tid, -1),
                    bbox=BoundingBox(x1, y1, x2, y2),
                    position_terrain=pos
                )

            joueurs_suivis.append(joueur_final)

        self.nb_joueurs_uniques = len(historique)
        return joueurs_suivis

    def _trouver_source(self, objets, cx: int, cy: int, seuil: int = 60):
        """Trouve l'objet brut le plus proche d'une position de track."""
        meilleur  = None
        dist_min  = float('inf')
        for o in objets:
            ocx = (o.bbox.x1 + o.bbox.x2) // 2
            ocy = (o.bbox.y1 + o.bbox.y2) // 2
            d   = np.sqrt((cx - ocx) ** 2 + (cy - ocy) ** 2)
            if d < dist_min and d < seuil:
                dist_min = d
                meilleur = o
        return meilleur

    # ─────────────────────────────────────
    # ACCÈS AUX DONNÉES
    # ─────────────────────────────────────

    def get_trajectoire(self, joueur_id: int) -> List[Position]:
        return self.historique_positions.get(joueur_id, [])

    def get_tous_ids(self) -> List[int]:
        return list(self.historique_positions.keys())

    def get_derniere_position(self, joueur_id: int) -> Optional[Position]:
        hist = self.historique_positions.get(joueur_id, [])
        return hist[-1] if hist else None

    def assigner_equipe(self, joueur_id: int, equipe_id: int):
        self.equipe_par_id[joueur_id] = equipe_id

    def get_joueurs_equipe(self, equipe_id: int) -> List[int]:
        return [jid for jid, eid in self.equipe_par_id.items()
                if eid == equipe_id]

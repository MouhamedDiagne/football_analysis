# services/s07_physique/physical_stats_service.py  — v2.0
# Stats physiques + analyse biomécanique (keypoints YOLOv8-Pose)

import numpy as np
from typing import List, Dict, Optional
from core.base_service import BaseService
from core.data_models import (
    Position, StatsPhysiquesJoueur, StatsBiomecaniques, PoseKeypoints
)


class PhysicalStatsService(BaseService):
    """
    Calcule toutes les statistiques physiques et biomécaniques
    des joueurs à partir de leurs trajectoires et de leurs poses.
    """

    # Seuils de vitesse (km/h)
    SEUIL_MARCHE       = 7.0
    SEUIL_TROT         = 14.0
    SEUIL_COURSE       = 21.0
    SEUIL_SPRINT       = 25.0
    DUREE_MIN_SPRINT_S = 1.0

    def __init__(self, fps: float = 25, pixels_par_metre: float = 10):
        super().__init__("PhysicalStats")
        self.fps             = fps
        self.pixels_par_metre = pixels_par_metre

        # Accumulation des poses par joueur pour la biomécanique
        # { joueur_id : [PoseKeypoints, ...] }
        self.historique_poses: Dict[int, List[PoseKeypoints]] = {}

    def initialiser(self):
        self.logger.info("✅ PhysicalStats (+ biomécanique) initialisé")
        self.est_initialise = True

    def traiter(self, data: dict) -> Dict[int, StatsPhysiquesJoueur]:
        return self.calculer_stats_tous_joueurs(data['historique_positions'])

    # ─────────────────────────────────────
    # ACCUMULATION DES POSES (appelé chaque frame)
    # ─────────────────────────────────────

    def accumuler_poses(self, poses: Dict[int, PoseKeypoints]):
        """Accumule les poses pour analyse biomécanique différée."""
        for jid, pose in poses.items():
            if jid not in self.historique_poses:
                self.historique_poses[jid] = []
            self.historique_poses[jid].append(pose)

    # ─────────────────────────────────────
    # STATS PHYSIQUES PAR JOUEUR
    # ─────────────────────────────────────

    def calculer_stats_joueur(self,
                               joueur_id: int,
                               trajectoire: List[Position]
                               ) -> StatsPhysiquesJoueur:
        stats = StatsPhysiquesJoueur(joueur_id=joueur_id)

        if len(trajectoire) < 2:
            return stats

        vitesses = self._calculer_serie_vitesses(trajectoire)
        if not vitesses:
            return stats

        # Distances par zone d'intensité
        distances = self._calculer_distances_par_zone(trajectoire, vitesses)
        stats.distance_marche_m  = distances['marche']
        stats.distance_trot_m    = distances['trot']
        stats.distance_course_m  = distances['course']
        stats.distance_sprint_m  = distances['sprint']
        stats.distance_totale_m  = sum(distances.values())

        # Vitesses
        vit = [v for v in vitesses if v > 0.5]
        if vit:
            stats.vitesse_max_kmh     = round(max(vit), 1)
            stats.vitesse_moyenne_kmh = round(np.mean(vit), 1)

        # Sprints
        sprints = self._analyser_sprints(trajectoire, vitesses)
        stats.nombre_sprints = len(sprints)

        # Accélérations
        stats.nombre_accelerations = self._compter_accelerations(vitesses)

        # Charge physique composite
        stats.charge_physique = self._calculer_charge(stats)

        # Mètres/minute
        duree_min = (trajectoire[-1].timestamp - trajectoire[0].timestamp) / 60
        if duree_min > 0:
            stats.metres_par_minute = round(
                stats.distance_totale_m / duree_min, 1
            )

        # Biomécanique (si poses disponibles)
        poses = self.historique_poses.get(joueur_id, [])
        if poses:
            bio = self._calculer_biomecanique(joueur_id, trajectoire, poses)
            stats.longueur_foulee_moy_m  = bio.longueur_foulee_moy_m
            stats.symetrie_course        = bio.symetrie_course
            stats.angle_inclination_moy  = bio.angle_inclination_moy

        return stats

    def calculer_stats_tous_joueurs(
            self,
            historique: Dict[int, List[Position]]
    ) -> Dict[int, StatsPhysiquesJoueur]:
        return {
            jid: self.calculer_stats_joueur(jid, traj)
            for jid, traj in historique.items()
        }

    # ─────────────────────────────────────
    # BIOMÉCANIQUE
    # ─────────────────────────────────────

    def calculer_biomecanique_tous(self) -> Dict[int, StatsBiomecaniques]:
        """Calcule les stats biomécaniques pour tous les joueurs avec poses."""
        resultats = {}
        for jid, poses in self.historique_poses.items():
            trajectoire = []   # Vide → OK, on se base sur les poses seules
            resultats[jid] = self._calculer_biomecanique(jid, trajectoire, poses)
        return resultats

    def _calculer_biomecanique(self,
                                joueur_id: int,
                                trajectoire: List[Position],
                                poses: List[PoseKeypoints]
                                ) -> StatsBiomecaniques:
        bio = StatsBiomecaniques(joueur_id=joueur_id)
        bio.nb_frames_pose = len(poses)

        if not poses:
            return bio

        # ── Inclinaison du tronc ──────────────────────────────────
        inclinaisons = [p.vecteur_inclination() for p in poses]
        inclinaisons = [a for a in inclinaisons if a is not None]
        if inclinaisons:
            bio.angle_inclination_moy = round(float(np.mean(inclinaisons)), 1)

        # ── Angles articulaires ───────────────────────────────────
        # Genou G : hanche_g(11) - genou_g(13) - cheville_g(15)
        # Genou D : hanche_d(12) - genou_d(14) - cheville_d(16)
        # Hanche G : epaule_g(5) - hanche_g(11) - genou_g(13)
        configs_angles = [
            ('angle_genou_g_moy',   11, 13, 15),
            ('angle_genou_d_moy',   12, 14, 16),
            ('angle_hanche_g_moy',   5, 11, 13),
            ('angle_hanche_d_moy',   6, 12, 14),
        ]
        for attr, ia, ib, ic in configs_angles:
            angles = [p.angle_articulation(ia, ib, ic) for p in poses]
            angles = [a for a in angles if a is not None]
            if angles:
                setattr(bio, attr, round(float(np.mean(angles)), 1))

        # ── Symétrie de course ────────────────────────────────────
        # Différence absolue entre angles genou G et D
        if bio.angle_genou_g_moy > 0 and bio.angle_genou_d_moy > 0:
            diff = abs(bio.angle_genou_g_moy - bio.angle_genou_d_moy)
            bio.symetrie_course = round(max(0.0, 100.0 - diff * 2), 1)

        # ── Amplitude des bras ────────────────────────────────────
        amplitudes = []
        for p in poses:
            pg = p.get_point_par_nom('poignet_g')
            pd = p.get_point_par_nom('poignet_d')
            if pg and pd:
                amplitudes.append(abs(pg[0] - pd[0]))
        if amplitudes:
            bio.amplitude_bras = round(float(np.mean(amplitudes)), 1)

        # ── Longueur de foulée estimée ────────────────────────────
        # Estimation depuis la trajectoire (si disponible)
        if len(trajectoire) >= 10 and self.pixels_par_metre > 0:
            longueurs = self._estimer_longueur_foulee(trajectoire, poses)
            if longueurs:
                bio.longueur_foulee_moy_m = round(float(np.mean(longueurs)), 2)

        # ── Charge articulaire estimée ────────────────────────────
        # Proportionnelle à la déviation par rapport à l'angle optimal
        OPTIMAL_GENOU  = 160.0   # degrés (course naturelle)
        OPTIMAL_HANCHE = 170.0
        if bio.angle_genou_g_moy > 0:
            bio.charge_genou = round(
                min(abs(OPTIMAL_GENOU - bio.angle_genou_g_moy) * 1.5, 100.0), 1
            )
        if bio.angle_hanche_g_moy > 0:
            bio.charge_hanche = round(
                min(abs(OPTIMAL_HANCHE - bio.angle_hanche_g_moy) * 1.5, 100.0), 1
            )

        # ── Fréquence de foulée ───────────────────────────────────
        if len(trajectoire) >= 4:
            bio.frequence_foulee_hz = self._estimer_frequence_foulee(
                trajectoire, poses
            )

        return bio

    def _estimer_longueur_foulee(self,
                                  trajectoire: List[Position],
                                  poses: List[PoseKeypoints]
                                  ) -> List[float]:
        """
        Estime la longueur de foulée via les phases de contact au sol
        (cheville alternativement en bas = contact).
        """
        longueurs = []
        N = min(len(trajectoire) - 1, len(poses) - 1)

        dernier_contact_pos = None
        for i in range(1, N):
            pose = poses[i]
            cg = pose.get_point_par_nom('cheville_g')
            cd = pose.get_point_par_nom('cheville_d')
            if cg is None and cd is None:
                continue

            # Contact au sol si cheville nettement plus basse que le centre de masse
            cm = pose.centre_masse()
            if cm is None:
                continue

            cheville_y = max(
                cg[1] if cg else 0,
                cd[1] if cd else 0
            )
            if cheville_y > cm[1] + 20:   # cheville clairement sous le CM
                pos = trajectoire[i]
                if dernier_contact_pos is not None:
                    dist_px = pos.distance_vers(dernier_contact_pos)
                    dist_m  = dist_px / max(self.pixels_par_metre, 1)
                    if 0.3 < dist_m < 3.0:   # foulée réaliste 0.3-3 m
                        longueurs.append(dist_m)
                dernier_contact_pos = pos

        return longueurs

    def _estimer_frequence_foulee(self,
                                   trajectoire: List[Position],
                                   poses: List[PoseKeypoints]
                                   ) -> float:
        """
        Estime la fréquence de foulée (Hz) via alternance des chevilles.
        """
        contacts = 0
        N = min(len(trajectoire), len(poses))
        cote_precedent = None

        for i in range(N):
            pose = poses[i]
            cg = pose.get_point_par_nom('cheville_g')
            cd = pose.get_point_par_nom('cheville_d')
            if cg is None or cd is None:
                continue

            # Quel côté est le plus bas (contact sol) ?
            cote = 'g' if cg[1] > cd[1] else 'd'
            if cote != cote_precedent and cote_precedent is not None:
                contacts += 1
            cote_precedent = cote

        duree_s = trajectoire[-1].timestamp - trajectoire[0].timestamp
        if duree_s > 0 and contacts > 0:
            return round(contacts / duree_s, 2)
        return 0.0

    # ─────────────────────────────────────
    # CALCUL DES VITESSES
    # ─────────────────────────────────────

    def _calculer_serie_vitesses(self,
                                  trajectoire: List[Position]
                                  ) -> List[float]:
        vitesses = []
        for i in range(1, len(trajectoire)):
            p1 = trajectoire[i - 1]
            p2 = trajectoire[i]
            dt = p2.timestamp - p1.timestamp
            if dt <= 0:
                vitesses.append(0.0)
                continue
            dist_px  = p1.distance_vers(p2)
            dist_m   = dist_px / max(self.pixels_par_metre, 1)
            vit_kmh  = (dist_m / dt) * 3.6
            vitesses.append(round(min(vit_kmh, 40.0), 2))
        return vitesses

    # ─────────────────────────────────────
    # DISTANCES PAR ZONE
    # ─────────────────────────────────────

    def _calculer_distances_par_zone(self,
                                      trajectoire: List[Position],
                                      vitesses: List[float]
                                      ) -> dict:
        d = {'marche': 0.0, 'trot': 0.0, 'course': 0.0, 'sprint': 0.0}
        for i, v in enumerate(vitesses):
            if i >= len(trajectoire) - 1:
                break
            dist_m = trajectoire[i].distance_vers(trajectoire[i + 1]) \
                     / max(self.pixels_par_metre, 1)
            if v < self.SEUIL_MARCHE:
                d['marche']  += dist_m
            elif v < self.SEUIL_TROT:
                d['trot']    += dist_m
            elif v < self.SEUIL_COURSE:
                d['course']  += dist_m
            else:
                d['sprint']  += dist_m
        return {k: round(v, 1) for k, v in d.items()}

    # ─────────────────────────────────────
    # SPRINTS
    # ─────────────────────────────────────

    def _analyser_sprints(self,
                           trajectoire: List[Position],
                           vitesses: List[float]) -> list:
        sprints     = []
        en_sprint   = False
        debut_idx   = None

        for i, v in enumerate(vitesses):
            if v >= self.SEUIL_SPRINT and not en_sprint:
                en_sprint = True
                debut_idx = i
            elif v < self.SEUIL_SPRINT and en_sprint:
                en_sprint = False
                if debut_idx is not None:
                    duree = (trajectoire[i].timestamp
                             - trajectoire[debut_idx].timestamp)
                    if duree >= self.DUREE_MIN_SPRINT_S:
                        vit_s = vitesses[debut_idx:i]
                        sprints.append({
                            'debut_ts': trajectoire[debut_idx].timestamp,
                            'fin_ts':   trajectoire[i].timestamp,
                            'duree_s':  round(duree, 2),
                            'vmax':     round(max(vit_s), 1),
                            'vmoy':     round(np.mean(vit_s), 1),
                        })
        return sprints

    # ─────────────────────────────────────
    # ACCÉLÉRATIONS
    # ─────────────────────────────────────

    def _compter_accelerations(self,
                                vitesses: List[float],
                                seuil: float = 3.0) -> int:
        return sum(
            1 for i in range(1, len(vitesses))
            if vitesses[i] - vitesses[i - 1] >= seuil
        )

    # ─────────────────────────────────────
    # CHARGE PHYSIQUE COMPOSITE
    # ─────────────────────────────────────

    def _calculer_charge(self, stats: StatsPhysiquesJoueur) -> float:
        score  = min(stats.distance_totale_m / 12000, 1.0) * 30
        score += min(stats.nombre_sprints      / 30, 1.0) * 25
        score += min(stats.distance_sprint_m   / 800, 1.0) * 25
        score += min(stats.vitesse_max_kmh     / 35, 1.0) * 20
        return round(score, 1)

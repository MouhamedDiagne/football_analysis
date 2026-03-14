# services/s09_analyse_match/match_analysis_service.py

import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from core.base_service import BaseService
from core.data_models import (
    Joueur, Ballon, Evenement, Position,
    StatsEquipe, StatsTechniquesJoueur,
    StatsPhysiquesJoueur, StatsTactiquesJoueur,
    StatsBiomecaniques,
    TypeEvenement, ZoneTerrain, Poste
)


# ─────────────────────────────────────────────────────────────
# RAPPORT COMPLET D'UN JOUEUR
# ─────────────────────────────────────────────────────────────

@dataclass
class RapportJoueur:
    joueur_id: int
    equipe_id: int
    poste: Poste = Poste.INCONNU
    # Physique
    distance_km: float = 0.0
    vitesse_max_kmh: float = 0.0
    vitesse_moyenne_kmh: float = 0.0
    # Technique
    passes_total: int = 0
    passes_reussies: int = 0
    passes_cles: int = 0
    taux_passes_pct: float = 0.0
    tirs_total: int = 0
    tirs_cadres: int = 0
    buts: int = 0
    dribbles_tentes: int = 0
    dribbles_reussis: int = 0
    # Duels
    duels_total: int = 0
    duels_gagnes: int = 0
    taux_duels_pct: float = 0.0
    # Defense
    interceptions: int = 0
    tacles: int = 0
    tacles_reussis: int = 0
    # Tactique
    heatmap: List[List[float]] = field(default_factory=list)
    zones_frequentees: List[str] = field(default_factory=list)
    position_moyenne: Tuple[float, float] = (0.0, 0.0)
    # Note de performance (0-10)
    note_performance: float = 0.0
    # Biomécanique v2
    longueur_foulee_moy_m:  float = 0.0
    symetrie_course:        float = 100.0
    angle_inclination_moy:  float = 0.0
    charge_genou:           float = 0.0
    charge_hanche:          float = 0.0


@dataclass
class RapportEquipe:
    equipe_id: int
    nom: str
    # Possession
    possession_pct: float = 0.0
    frames_possession: int = 0
    # Tirs
    tirs_total: int = 0
    tirs_cadres: int = 0
    tirs_non_cadres: int = 0
    buts: int = 0
    expected_goals: float = 0.0
    # Passes
    passes_total: int = 0
    passes_reussies: int = 0
    passes_ratees: int = 0
    taux_passes_pct: float = 0.0
    passes_progressives: int = 0
    # Events
    corners: int = 0
    fautes: int = 0
    hors_jeux: int = 0
    cartons_jaunes: int = 0
    cartons_rouges: int = 0
    # Pressing
    pressing_intensite: float = 0.0
    pressing_score_cumule: float = 0.0
    pressing_frames: int = 0
    ppda: float = 0.0
    # Zones de recuperation
    zones_recuperation: Dict[str, int] = field(default_factory=dict)
    # Tactique
    formation: str = ""
    bloc_defensif_hauteur: float = 0.0
    largeur_jeu: float = 0.0
    # Distance collective
    distance_totale_km: float = 0.0


@dataclass
class RapportMatch:
    duree_s: float = 0.0
    equipes: Dict[int, RapportEquipe] = field(default_factory=dict)
    joueurs: Dict[int, RapportJoueur] = field(default_factory=dict)
    nb_frames_traitees: int = 0
    evenements_total: int = 0


# ─────────────────────────────────────────────────────────────
# SERVICE PRINCIPAL
# ─────────────────────────────────────────────────────────────

class MatchAnalysisService(BaseService):
    """
    Service metier principal d'analyse video de match de football.

    Agregation de toutes les statistiques :
    - Equipe  : possession, tirs, passes, corners, fautes, hors-jeux,
                xG, pressing, zones de recuperation, formation
    - Joueur  : distance, vitesse, duels, passes cles, dribbles,
                interceptions, tacles, heatmap, note de performance
    """

    # Dimensions du terrain (metres)
    TERRAIN_L = 105.0
    TERRAIN_W = 68.0

    # Grille heatmap
    HEATMAP_GRILLE = 12

    # Seuil pressing (metres)
    SEUIL_PRESSING_M = 5.0

    # Seuil passe progressive (avance d'au moins N metres vers but adverse)
    SEUIL_PASSE_PROGRESSIVE_M = 10.0

    # Seuil passe cle (zone de tir < 25m du but)
    SEUIL_PASSE_CLE_M = 25.0

    def __init__(self,
                 nom_equipe_0: str = "Equipe A",
                 nom_equipe_1: str = "Equipe B",
                 fps: float = 25.0,
                 pixels_par_metre: float = 10.0):
        super().__init__("MatchAnalysis")
        self.fps = fps
        self.pixels_par_metre = pixels_par_metre

        # Rapports courants
        self._rapport = RapportMatch()
        self._rapport.equipes[0] = RapportEquipe(0, nom_equipe_0)
        self._rapport.equipes[1] = RapportEquipe(1, nom_equipe_1)

        # Accumulation possession frames
        self._frames_possession_total: int = 0

        # Historique positions par joueur (pour heatmap et positions moy.)
        self._positions_acc: Dict[int, List[Tuple[float, float]]] = {}

        # Mapping joueur -> equipe
        self._joueur_equipe: Dict[int, int] = {}
        self._joueur_poste: Dict[int, Poste] = {}

        # Pressing : cumul scores par frame
        self._pressing_cumul: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._pressing_frames: int = 0

    # ─────────────────────────────────────
    # INITIALISATION / TRAITEMENT
    # ─────────────────────────────────────

    def initialiser(self):
        self.logger.info("MatchAnalysis initialise")
        self.est_initialise = True

    def traiter(self, data: dict) -> dict:
        """
        Point d'entree par frame.

        data attendu :
            joueurs   : List[Joueur]
            ballon    : Optional[Ballon]
            evenements: List[Evenement]
            frame_id  : int
            timestamp : float
            pressing_frame: Optional[Dict[int, float]]
            formation : Optional[Dict[int, str]]
        """
        joueurs    = data.get('joueurs', [])
        ballon     = data.get('ballon')
        evenements = data.get('evenements', [])
        frame_id   = data.get('frame_id', 0)
        timestamp  = data.get('timestamp', 0.0)
        pressing   = data.get('pressing_frame', {})
        formation  = data.get('formation', {})

        self._traiter_frame(
            joueurs, ballon, evenements,
            frame_id, timestamp, pressing, formation
        )
        self._rapport.nb_frames_traitees += 1
        self._rapport.duree_s = timestamp

        return {'status': 'ok', 'frame_id': frame_id}

    # ─────────────────────────────────────
    # TRAITEMENT PAR FRAME
    # ─────────────────────────────────────

    def _traiter_frame(self,
                       joueurs: List[Joueur],
                       ballon: Optional[Ballon],
                       evenements: List[Evenement],
                       frame_id: int,
                       timestamp: float,
                       pressing_frame: Dict[int, float],
                       formation: Dict[int, str]):

        # 1. Enregistrer les joueurs detectes
        self._enregistrer_joueurs(joueurs, timestamp)

        # 2. Possession du ballon
        if ballon is not None:
            self._maj_possession(ballon)

        # 3. Traiter chaque evenement
        for evt in evenements:
            self._traiter_evenement(evt)
            self._rapport.evenements_total += 1

        # 4. Pressing
        self._maj_pressing(pressing_frame)

        # 5. Formation
        for eq_id, form in formation.items():
            if form and form != "N/A":
                self._rapport.equipes[eq_id].formation = form

    # ─────────────────────────────────────
    # ENREGISTREMENT JOUEURS
    # ─────────────────────────────────────

    def _enregistrer_joueurs(self,
                              joueurs: List[Joueur],
                              timestamp: float):
        for j in joueurs:
            jid = j.id
            eq_id = j.equipe_id

            self._joueur_equipe[jid] = eq_id
            self._joueur_poste[jid] = j.poste

            # Assurer l'existence du rapport joueur
            if jid not in self._rapport.joueurs:
                self._rapport.joueurs[jid] = RapportJoueur(
                    joueur_id=jid,
                    equipe_id=eq_id,
                    poste=j.poste
                )

            # Accumuler position (en metres)
            x_m = j.position_terrain.x / self.pixels_par_metre
            y_m = j.position_terrain.y / self.pixels_par_metre
            if jid not in self._positions_acc:
                self._positions_acc[jid] = []
            self._positions_acc[jid].append((x_m, y_m))

    # ─────────────────────────────────────
    # POSSESSION
    # ─────────────────────────────────────

    def _maj_possession(self, ballon: Ballon):
        eq = ballon.equipe_possesseur
        if eq is not None and eq in self._rapport.equipes:
            self._rapport.equipes[eq].frames_possession += 1
        self._frames_possession_total += 1

    def _calculer_possession(self):
        total = self._frames_possession_total
        if total == 0:
            return
        for eq_id, eq in self._rapport.equipes.items():
            eq.possession_pct = round(
                (eq.frames_possession / total) * 100, 1
            )

    # ─────────────────────────────────────
    # EVENEMENTS
    # ─────────────────────────────────────

    def _traiter_evenement(self, evt: Evenement):
        eq_id = evt.equipe_id
        jid   = evt.joueur_id
        eq    = self._rapport.equipes.get(eq_id) if eq_id is not None else None
        jp    = self._rapport.joueurs.get(jid) if jid is not None else None

        t = evt.type

        if t == TypeEvenement.TIR:
            self._evt_tir(eq, jp, evt)

        elif t == TypeEvenement.PASSE:
            self._evt_passe(eq, jp, evt)

        elif t == TypeEvenement.CORNER:
            if eq:
                eq.corners += 1

        elif t == TypeEvenement.FAUTE:
            self._evt_faute(eq, jp, evt)

        elif t == TypeEvenement.HORS_JEU:
            if eq:
                eq.hors_jeux += 1

        elif t == TypeEvenement.BUT:
            if eq:
                eq.buts += 1
            if jp:
                jp.buts += 1

        elif t == TypeEvenement.DRIBBLE:
            if jp:
                jp.dribbles_tentes += 1
                if evt.reussi:
                    jp.dribbles_reussis += 1

        elif t == TypeEvenement.TACLE:
            if jp:
                jp.tacles += 1
                if evt.reussi:
                    jp.tacles_reussis += 1

        elif t == TypeEvenement.INTERCEPTION:
            if jp:
                jp.interceptions += 1
            # Zone de recuperation
            if eq and eq_id is not None:
                zone = self._zone_depuis_position(evt.position, eq_id)
                eq.zones_recuperation[zone] = (
                    eq.zones_recuperation.get(zone, 0) + 1
                )

    def _evt_tir(self,
                 eq: Optional[RapportEquipe],
                 jp: Optional[RapportJoueur],
                 evt: Evenement):
        if eq:
            eq.tirs_total += 1
            # Utiliser le xG calculé par S8 si disponible, sinon recalculer
            xg = evt.valeur_xg if evt.valeur_xg > 0 else self._calculer_xg(evt.position, evt.equipe_id)
            eq.expected_goals = round(eq.expected_goals + xg, 3)
            if evt.reussi:
                eq.tirs_cadres += 1
            else:
                eq.tirs_non_cadres += 1
        if jp:
            jp.tirs_total += 1
            jp.tirs_cadres += 1 if evt.reussi else 0

    def _evt_passe(self,
                   eq: Optional[RapportEquipe],
                   jp: Optional[RapportJoueur],
                   evt: Evenement):
        if eq:
            eq.passes_total += 1
            if evt.reussi:
                eq.passes_reussies += 1
                if self._est_passe_progressive(evt):
                    eq.passes_progressives += 1
            else:
                eq.passes_ratees += 1
        if jp:
            jp.passes_total += 1
            if evt.reussi:
                jp.passes_reussies += 1
                if self._est_passe_cle(evt):
                    jp.passes_cles += 1

    def _evt_faute(self,
                   eq: Optional[RapportEquipe],
                   jp: Optional[RapportJoueur],
                   evt: Evenement):
        if eq:
            eq.fautes += 1

    # ─────────────────────────────────────
    # PRESSING
    # ─────────────────────────────────────

    def _maj_pressing(self, pressing_frame: Dict[int, float]):
        if not pressing_frame:
            return
        self._pressing_frames += 1
        for eq_id, score in pressing_frame.items():
            if eq_id in self._pressing_cumul:
                self._pressing_cumul[eq_id] += score

    def _calculer_pressing_final(self):
        if self._pressing_frames == 0:
            return
        for eq_id, eq in self._rapport.equipes.items():
            eq.pressing_score_cumule = round(
                self._pressing_cumul.get(eq_id, 0.0), 2
            )
            eq.pressing_intensite = round(
                self._pressing_cumul.get(eq_id, 0.0)
                / self._pressing_frames, 3
            )
            eq.pressing_frames = self._pressing_frames

    # ─────────────────────────────────────
    # PPDA (Passes Per Defensive Action)
    # ─────────────────────────────────────

    def _calculer_ppda(self):
        """
        PPDA = passes adverses autorisees / actions defensives propres
        Mesure la pression haute : valeur basse = pressing intense
        """
        for eq_id, eq in self._rapport.equipes.items():
            eq_adv_id = 1 - eq_id
            eq_adv = self._rapport.equipes.get(eq_adv_id)
            if eq_adv is None:
                continue

            # Passes adverses dans la moitie defensive propre
            passes_adv = eq_adv.passes_total
            # Actions defensives propres
            actions_def = sum(
                jp.interceptions + jp.tacles
                for jid, jp in self._rapport.joueurs.items()
                if jp.equipe_id == eq_id
            )
            if actions_def > 0:
                eq.ppda = round(passes_adv / actions_def, 2)
            else:
                eq.ppda = 99.0

    # ─────────────────────────────────────
    # HEATMAP & STATISTIQUES TACTIQUES
    # ─────────────────────────────────────

    def _generer_heatmap_joueur(self, joueur_id: int) -> List[List[float]]:
        positions = self._positions_acc.get(joueur_id, [])
        if not positions:
            return []

        grille = self.HEATMAP_GRILLE
        heatmap = np.zeros((grille, grille))

        for x, y in positions:
            col = int(max(0, min((x / self.TERRAIN_L) * grille, grille - 1)))
            row = int(max(0, min((y / self.TERRAIN_W) * grille, grille - 1)))
            heatmap[row][col] += 1

        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max()) * 100

        return np.round(heatmap, 1).tolist()

    def _calculer_stats_tactiques_joueur(self,
                                          joueur_id: int,
                                          jp: RapportJoueur):
        positions = self._positions_acc.get(joueur_id, [])
        if not positions:
            return

        arr = np.array(positions)
        jp.position_moyenne = (
            round(float(np.mean(arr[:, 0])), 1),
            round(float(np.mean(arr[:, 1])), 1)
        )
        jp.heatmap = self._generer_heatmap_joueur(joueur_id)
        jp.zones_frequentees = self._identifier_zones(positions, jp.equipe_id)

    def _identifier_zones(self,
                           positions: List[Tuple[float, float]],
                           equipe_id: int) -> List[str]:
        """Identifier les 3 zones les plus frequentees"""
        grille_zones = {
            "Defence profonde": 0,
            "Defence": 0,
            "Milieu defensif": 0,
            "Milieu central": 0,
            "Milieu offensif": 0,
            "Attaque": 0,
            "Surface de reparation": 0
        }
        for x, y in positions:
            zone = self._zone_depuis_xy(x, y, equipe_id)
            grille_zones[zone] = grille_zones.get(zone, 0) + 1

        triees = sorted(grille_zones.items(), key=lambda kv: kv[1], reverse=True)
        return [z for z, cnt in triees[:3] if cnt > 0]

    # ─────────────────────────────────────
    # STATISTIQUES PHYSIQUES
    # ─────────────────────────────────────

    def integrer_stats_biomecaniques(self,
                                      stats_bio: Dict[int, StatsBiomecaniques]):
        """Intègre les stats biomécaniques dans le rapport joueur."""
        for jid, bio in stats_bio.items():
            jp = self._rapport.joueurs.get(jid)
            if jp is None:
                continue
            jp.longueur_foulee_moy_m = bio.longueur_foulee_moy_m
            jp.symetrie_course       = bio.symetrie_course
            jp.angle_inclination_moy = bio.angle_inclination_moy
            jp.charge_genou          = bio.charge_genou
            jp.charge_hanche         = bio.charge_hanche

    def integrer_stats_physiques(self,
                                  stats_physiques: Dict[int, StatsPhysiquesJoueur]):
        """
        Integrer les stats du PhysicalStatsService dans le rapport.
        A appeler apres la fin du match.
        """
        for jid, sp in stats_physiques.items():
            jp = self._rapport.joueurs.get(jid)
            if jp is None:
                continue
            jp.distance_km = round(sp.distance_totale_m / 1000, 2)
            jp.vitesse_max_kmh = sp.vitesse_max_kmh
            jp.vitesse_moyenne_kmh = sp.vitesse_moyenne_kmh

            # Ajouter au total equipe
            eq_id = jp.equipe_id
            eq = self._rapport.equipes.get(eq_id)
            if eq:
                eq.distance_totale_km = round(
                    eq.distance_totale_km + jp.distance_km, 2
                )

    # ─────────────────────────────────────
    # NOTE DE PERFORMANCE JOUEUR
    # ─────────────────────────────────────

    def _calculer_note_performance(self, jp: RapportJoueur) -> float:
        """
        Note composite 0-10 basee sur 5 dimensions :
        - Physique     (20 %)
        - Technique    (30 %)
        - Duels        (20 %)
        - Defense      (15 %)
        - Contribution (15 %)
        """
        score = 0.0

        # -- Physique (0-2) --
        dist_score = min(jp.distance_km / 12.0, 1.0)
        vit_score  = min(jp.vitesse_max_kmh / 35.0, 1.0)
        score += (dist_score * 0.6 + vit_score * 0.4) * 2.0

        # -- Technique (0-3) --
        pass_pct = (jp.passes_reussies / jp.passes_total
                    if jp.passes_total > 0 else 0)
        drib_pct = (jp.dribbles_reussis / jp.dribbles_tentes
                    if jp.dribbles_tentes > 0 else 0)
        passes_cles_score = min(jp.passes_cles / 5.0, 1.0)
        score += (pass_pct * 0.5 + drib_pct * 0.3 + passes_cles_score * 0.2) * 3.0

        # -- Duels (0-2) --
        duel_pct = (jp.duels_gagnes / jp.duels_total
                    if jp.duels_total > 0 else 0)
        score += duel_pct * 2.0

        # -- Defense (0-1.5) --
        def_score = min((jp.interceptions + jp.tacles_reussis) / 8.0, 1.0)
        score += def_score * 1.5

        # -- Contribution offensive (0-1.5) --
        offens_score = min(
            (jp.buts * 3 + jp.passes_cles + jp.tirs_cadres) / 10.0, 1.0
        )
        score += offens_score * 1.5

        return round(min(score, 10.0), 1)

    # ─────────────────────────────────────
    # RAPPORT FINAL
    # ─────────────────────────────────────

    def generer_rapport(self) -> RapportMatch:
        """
        Finaliser et retourner le rapport complet du match.
        A appeler en fin de traitement video.
        """
        self.logger.info("Generation du rapport final...")

        # Possession
        self._calculer_possession()

        # Taux passes equipes
        for eq in self._rapport.equipes.values():
            if eq.passes_total > 0:
                eq.taux_passes_pct = round(
                    (eq.passes_reussies / eq.passes_total) * 100, 1
                )

        # Pressing final
        self._calculer_pressing_final()

        # PPDA
        self._calculer_ppda()

        # Stats tactiques et notes joueurs
        for jid, jp in self._rapport.joueurs.items():
            self._calculer_stats_tactiques_joueur(jid, jp)

            if jp.passes_total > 0:
                jp.taux_passes_pct = round(
                    (jp.passes_reussies / jp.passes_total) * 100, 1
                )
            if jp.duels_total > 0:
                jp.taux_duels_pct = round(
                    (jp.duels_gagnes / jp.duels_total) * 100, 1
                )

            jp.note_performance = self._calculer_note_performance(jp)

        # Bloc defensif et largeur de jeu
        self._calculer_dimensions_jeu()

        self.logger.info(
            f"Rapport genere : {len(self._rapport.joueurs)} joueurs, "
            f"{self._rapport.evenements_total} evenements"
        )
        return self._rapport

    # ─────────────────────────────────────
    # INTEGRATION STATS EXTERIEURES
    # ─────────────────────────────────────

    def enregistrer_duel(self,
                          joueur_attaquant_id: int,
                          joueur_defenseur_id: int,
                          gagnant_id: int):
        """Enregistrer le resultat d'un duel"""
        for jid in [joueur_attaquant_id, joueur_defenseur_id]:
            jp = self._rapport.joueurs.get(jid)
            if jp:
                jp.duels_total += 1
                if jid == gagnant_id:
                    jp.duels_gagnes += 1

    # ─────────────────────────────────────
    # METHODES UTILITAIRES
    # ─────────────────────────────────────

    def _calculer_xg(self,
                     position: Position,
                     equipe_id: Optional[int]) -> float:
        """
        Modele xG simplifie base sur la distance et l'angle au but.
        But adverse : x=105 pour eq_id=0, x=0 pour eq_id=1
        """
        x_m = position.x / self.pixels_par_metre
        y_m = position.y / self.pixels_par_metre

        # Position du but adverse
        bx = self.TERRAIN_L if equipe_id == 0 else 0.0
        by = self.TERRAIN_W / 2

        dist = math.sqrt((x_m - bx) ** 2 + (y_m - by) ** 2)

        # Angle vers les poteaux (poteaux a +/-3.66m du centre)
        if dist == 0:
            return 1.0

        angle = math.degrees(math.atan2(7.32 / 2, dist))

        # Modele logistique empirique
        xg = (angle / 45.0) * math.exp(-dist / 20.0)
        return round(min(xg, 1.0), 4)

    def _est_passe_progressive(self, evt: Evenement) -> bool:
        """Passe qui avance le ballon vers le but adverse"""
        if evt.position_destination is None:
            return False
        x_src  = evt.position.x / self.pixels_par_metre
        x_dest = evt.position_destination.x / self.pixels_par_metre
        if evt.equipe_id == 0:
            return (x_dest - x_src) >= self.SEUIL_PASSE_PROGRESSIVE_M
        else:
            return (x_src - x_dest) >= self.SEUIL_PASSE_PROGRESSIVE_M

    def _est_passe_cle(self, evt: Evenement) -> bool:
        """Passe dont la destination est dans la zone de tir"""
        if evt.position_destination is None:
            return False
        x_dest = evt.position_destination.x / self.pixels_par_metre
        y_dest = evt.position_destination.y / self.pixels_par_metre
        bx = self.TERRAIN_L if evt.equipe_id == 0 else 0.0
        by = self.TERRAIN_W / 2
        dist = math.sqrt((x_dest - bx) ** 2 + (y_dest - by) ** 2)
        return dist <= self.SEUIL_PASSE_CLE_M

    def _zone_depuis_position(self,
                               position: Position,
                               equipe_id: int) -> str:
        x_m = position.x / self.pixels_par_metre
        y_m = position.y / self.pixels_par_metre
        return self._zone_depuis_xy(x_m, y_m, equipe_id)

    def _zone_depuis_xy(self,
                        x: float,
                        y: float,
                        equipe_id: int) -> str:
        """Determiner la zone de terrain depuis les coordonnees en metres"""
        # Normaliser selon sens d'attaque
        if equipe_id == 1:
            x = self.TERRAIN_L - x

        if x < 17:
            return "Defence profonde"
        elif x < 35:
            return "Defence"
        elif x < 52.5:
            return "Milieu defensif"
        elif x < 70:
            return "Milieu central"
        elif x < 87:
            return "Milieu offensif"
        else:
            # Verifier si dans la surface
            if abs(y - self.TERRAIN_W / 2) <= 20.16 and x >= (self.TERRAIN_L - 16.5):
                return "Surface de reparation"
            return "Attaque"

    def _calculer_dimensions_jeu(self):
        """Calcul hauteur bloc defensif et largeur de jeu par equipe"""
        for eq_id, eq in self._rapport.equipes.items():
            positions_eq = [
                pos
                for jid, positions in self._positions_acc.items()
                if self._joueur_equipe.get(jid) == eq_id
                and self._joueur_poste.get(jid) != Poste.GARDIEN
                for pos in positions
            ]
            if not positions_eq:
                continue
            arr = np.array(positions_eq)
            # Hauteur bloc : percentile 10 sur axe X
            eq.bloc_defensif_hauteur = round(
                float(np.percentile(arr[:, 0], 10)), 1
            )
            # Largeur jeu : ecart-type sur axe Y * 2
            eq.largeur_jeu = round(
                float(np.percentile(arr[:, 1], 90)
                      - np.percentile(arr[:, 1], 10)), 1
            )

    def get_resume_equipe(self, equipe_id: int) -> dict:
        """Retourner un dictionnaire lisible des stats equipe"""
        eq = self._rapport.equipes.get(equipe_id)
        if eq is None:
            return {}
        return {
            "Equipe": eq.nom,
            "Possession (%)": eq.possession_pct,
            "Tirs (cadres / total)": f"{eq.tirs_cadres} / {eq.tirs_total}",
            "Expected Goals (xG)": eq.expected_goals,
            "Passes (%)": f"{eq.passes_reussies}/{eq.passes_total} ({eq.taux_passes_pct}%)",
            "Passes progressives": eq.passes_progressives,
            "Corners": eq.corners,
            "Fautes": eq.fautes,
            "Hors-jeux": eq.hors_jeux,
            "Pressing intensite": eq.pressing_intensite,
            "PPDA": eq.ppda,
            "Zones de recuperation": eq.zones_recuperation,
            "Formation": eq.formation,
            "Distance totale (km)": eq.distance_totale_km,
            "Bloc defensif hauteur (m)": eq.bloc_defensif_hauteur,
            "Largeur de jeu (m)": eq.largeur_jeu,
        }

    def get_resume_joueur(self, joueur_id: int) -> dict:
        """Retourner un dictionnaire lisible des stats joueur"""
        jp = self._rapport.joueurs.get(joueur_id)
        if jp is None:
            return {}
        return {
            "ID": jp.joueur_id,
            "Equipe": jp.equipe_id,
            "Poste": jp.poste.value,
            "Distance (km)": jp.distance_km,
            "Vitesse max (km/h)": jp.vitesse_max_kmh,
            "Vitesse moyenne (km/h)": jp.vitesse_moyenne_kmh,
            "Duels (gagnes/total)": f"{jp.duels_gagnes}/{jp.duels_total} ({jp.taux_duels_pct}%)",
            "Passes cles": jp.passes_cles,
            "Passes (%)": f"{jp.passes_reussies}/{jp.passes_total} ({jp.taux_passes_pct}%)",
            "Dribbles (reussis/tentes)": f"{jp.dribbles_reussis}/{jp.dribbles_tentes}",
            "Interceptions": jp.interceptions,
            "Tacles (reussis/total)": f"{jp.tacles_reussis}/{jp.tacles}",
            "Position moyenne (m)": jp.position_moyenne,
            "Zones frequentees": jp.zones_frequentees,
            "Note de performance": jp.note_performance,
            # Biomécanique v2
            "Longueur foulée moy (m)": jp.longueur_foulee_moy_m,
            "Symétrie course (%)": jp.symetrie_course,
            "Inclinaison tronc (°)": jp.angle_inclination_moy,
        }

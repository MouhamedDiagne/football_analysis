# main.py  — v2.0
# Pipeline complet d'analyse vidéo de match de football
#
# Ordre d'exécution :
#   S1   VideoIngestion      → charger vidéo, générer frames
#   S2   TerrainDetection    → homographie, calibration px/m
#   S3   PlayerDetection     → YOLOv9c : joueurs + ballon + arbitres
#   S3b  PoseEstimation      → YOLOv8-Pose : 17 keypoints COCO
#   S4   Tracking            → ByteTrack : ID persistant
#   S5   TeamClassification  → ResNet-18 : équipe 0 / équipe 1
#   S6   BallTracking        → tracking ballon + possession
#   S7   PhysicalStats       → distances, vitesses, biomécanique
#   S8   TacticalAnalysis    → heatmaps, passes, pressing Voronoi, xG
#   S9   MatchAnalysis       → rapport final équipes + joueurs

import sys
import json
import logging

from config import settings
from core.data_models import Evenement, TypeEvenement, Position

from services.s01_ingestion.video_ingestion_service    import VideoIngestionService
from services.s02_terrain.terrain_detection_service    import TerrainDetectionService
from services.s03_detection.player_detection_service   import PlayerDetectionService
from services.s03b_pose.pose_estimation_service        import PoseEstimationService
from services.s04_tracking.tracking_service            import TrackingService
from services.s05_equipes.team_classification_service  import TeamClassificationService
from services.s06_ballon.ball_tracking_service         import BallTrackingService
from services.s07_physique.physical_stats_service      import PhysicalStatsService
from services.s08_tactique.tactical_analysis_service   import TacticalAnalysisService
from services.s09_analyse_match.match_analysis_service import MatchAnalysisService

logging.basicConfig(level=logging.INFO)
settings.configure_runtime_environment()


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

CHEMIN_VIDEO     = "video/match_wfc_Sonacos.mp4"
NOM_EQUIPE_0     = "WFC"
NOM_EQUIPE_1     = "SONACOS"
MODELE_DETECTION = "yolov9c.pt"      # détection joueurs/ballon
MODELE_POSE      = "yolov8n-pose.pt" # estimation pose


# ─────────────────────────────────────────────────────────────
# STEP 0 — Instanciation des services
# ─────────────────────────────────────────────────────────────

def creer_services():
    print("\n[STEP 0] Instanciation des services...")

    s1_ingestion  = VideoIngestionService()
    s2_terrain    = TerrainDetectionService()
    s3_detection  = PlayerDetectionService(model_path=MODELE_DETECTION)
    s3b_pose      = PoseEstimationService(model_path=MODELE_POSE)
    s4_tracking   = TrackingService()
    s5_equipes    = TeamClassificationService()
    s6_ballon     = BallTrackingService()
    s7_physique   = PhysicalStatsService()
    s8_tactique   = TacticalAnalysisService()
    s9_analyse    = MatchAnalysisService(
        nom_equipe_0=NOM_EQUIPE_0,
        nom_equipe_1=NOM_EQUIPE_1
    )

    return (s1_ingestion, s2_terrain, s3_detection, s3b_pose,
            s4_tracking, s5_equipes, s6_ballon,
            s7_physique, s8_tactique, s9_analyse)


# ─────────────────────────────────────────────────────────────
# STEP 1 — Initialisation
# ─────────────────────────────────────────────────────────────

def initialiser_services(services: tuple):
    print("\n[STEP 1] Initialisation des services...")
    for svc in services:
        svc.initialiser()
    print("  Tous les services sont prêts.\n")


# ─────────────────────────────────────────────────────────────
# STEP 2 — Chargement vidéo
# ─────────────────────────────────────────────────────────────

def charger_video(s1: VideoIngestionService):
    print(f"\n[STEP 2] Chargement : {CHEMIN_VIDEO}")
    meta = s1.charger_video(CHEMIN_VIDEO)
    print(f"  Fichier   : {meta.nom_fichier}")
    print(f"  FPS       : {meta.fps}")
    print(f"  Frames    : {meta.total_frames}")
    print(f"  Durée     : {meta.duree_minutes:.1f} min")
    print(f"  Résolution: {meta.largeur}×{meta.hauteur} ({meta.resolutions_standard})")
    return meta


# ─────────────────────────────────────────────────────────────
# STEP 3 — Calibration terrain
# ─────────────────────────────────────────────────────────────

def calibrer_terrain(s1: VideoIngestionService,
                     s2: TerrainDetectionService):
    print("\n[STEP 3] Calibration terrain...")
    premiere = s1.extraire_frame(0)
    if premiere is None:
        print("  ERREUR : impossible de lire la frame 0")
        return
    res = s2.analyser_terrain(premiere)
    if res.get('homographie') is not None:
        print(f"  Homographie calculée — {s2.pixels_par_metre_x:.1f} px/m")
    else:
        print("  AVERTISSEMENT : homographie absente (mode fallback)")


# ─────────────────────────────────────────────────────────────
# STEP 4 — Boucle principale frame par frame
# ─────────────────────────────────────────────────────────────

def boucle_frames(services: tuple, fps: float, max_frames: int = 0):
    print("\n[STEP 4] Traitement frame par frame...")

    (s1, s2, s3, s3b, s4, s5, s6, s7, s8, s9) = services

    # Propagation fps et calibration
    s4.fps              = int(fps)
    s6.fps              = fps
    s7.fps              = fps
    s8.fps              = fps
    s9.fps              = fps
    s6.pixels_par_metre = s2.pixels_par_metre_x
    s7.pixels_par_metre = s2.pixels_par_metre_x
    s8.pixels_par_metre = s2.pixels_par_metre_x
    s9.pixels_par_metre = s2.pixels_par_metre_x

    s5.definir_noms_equipes(NOM_EQUIPE_0, NOM_EQUIPE_1)

    nb_frames = 0

    for frame, frame_id, timestamp in s1.generer_frames():
        if max_frames > 0 and nb_frames >= max_frames:
            break

        # ── S3 : Détection YOLOv9c ───────────────────────────────
        res_det    = s3.detecter(frame, frame_id, timestamp)
        joueurs_br = res_det['joueurs']
        arbitres_br = res_det['arbitres']
        ballon_br  = res_det['ballon']

        # ── S4 : Tracking ByteTrack ──────────────────────────────
        joueurs_suivis  = s4.mettre_a_jour(frame, joueurs_br, timestamp, frame_id)
        arbitres_suivis = s4.mettre_a_jour_arbitres(arbitres_br, timestamp, frame_id)

        # ── S3b : Pose Estimation YOLOv8-Pose ────────────────────
        poses = s3b.estimer_poses(frame, joueurs_suivis, frame_id)
        s7.accumuler_poses(poses)   # pour la biomécanique différée

        # ── S5 : Classification équipes ResNet-18 ────────────────
        joueurs_classes = s5.classifier_tous(frame, joueurs_suivis)
        for j in joueurs_classes:
            s4.assigner_equipe(j.id, j.equipe_id)

        # ── S6 : Tracking ballon + possession ────────────────────
        res_ballon = s6.mettre_a_jour(
            ballon_br, joueurs_classes, timestamp, frame_id
        )
        ballon = res_ballon.get('ballon')

        # ── S8 : Analyse tactique (passes, pressing, xG) ─────────
        res_tactique = s8.analyser_frame(
            joueurs_classes, ballon, frame_id, timestamp
        )
        pressing_frame  = res_tactique.get('pressing', {})
        formation       = res_tactique.get('formation', {})
        evts_frame      = res_tactique.get('evenements', [])

        # ── S9 : Accumulation rapport match ──────────────────────
        s9.traiter({
            'joueurs'       : joueurs_classes,
            'ballon'        : ballon,
            'evenements'    : evts_frame,
            'frame_id'      : frame_id,
            'timestamp'     : timestamp,
            'pressing_frame': pressing_frame,
            'formation'     : formation,
        })

        nb_frames += 1

        if nb_frames % 250 == 0:
            print(f"  Frame {frame_id:5d} | t={timestamp:6.1f}s "
                  f"| joueurs={len(joueurs_classes):2d} "
                  f"| arbitres={len(arbitres_suivis):1d} "
                  f"| poses={len(poses):2d}")

    print(f"\n  Boucle terminée : {nb_frames} frames traitées.")
    return nb_frames


# ─────────────────────────────────────────────────────────────
# STEP 5 — Stats physiques + biomécanique finales
# ─────────────────────────────────────────────────────────────

def calculer_stats_physiques(s4: TrackingService,
                              s7: PhysicalStatsService,
                              s9: MatchAnalysisService):
    print("\n[STEP 5] Stats physiques & biomécaniques finales...")

    stats_phys = s7.calculer_stats_tous_joueurs(s4.historique_positions)
    s9.integrer_stats_physiques(stats_phys)

    stats_bio  = s7.calculer_biomecanique_tous()
    s9.integrer_stats_biomecaniques(stats_bio)

    print(f"  Stats physiques  : {len(stats_phys)} joueurs")
    print(f"  Stats biomécanique: {len(stats_bio)} joueurs")
    return stats_phys, stats_bio


# ─────────────────────────────────────────────────────────────
# STEP 6 — Rapport final
# ─────────────────────────────────────────────────────────────

def generer_rapport(s9: MatchAnalysisService):
    print("\n[STEP 6] Génération du rapport final...")

    rapport = s9.generer_rapport()

    print("\n" + "=" * 60)
    print("  RAPPORT MATCH")
    print("=" * 60)
    print(f"  Durée analysée : {rapport.duree_s:.0f}s")
    print(f"  Frames traitées: {rapport.nb_frames_traitees}")
    print(f"  Événements     : {rapport.evenements_total}")
    print(f"  Joueurs suivis : {len(rapport.joueurs)}")

    for eq_id in [0, 1]:
        print(f"\n  {'─'*55}")
        resume = s9.get_resume_equipe(eq_id)
        print(f"  {resume.get('Equipe', f'Équipe {eq_id}')}")
        for cle, val in list(resume.items())[:8]:
            print(f"    {cle:<35}: {val}")

    print(f"\n  {'─'*55}")
    print("  TOP JOUEURS (note de performance)")
    classement = sorted(
        rapport.joueurs.values(),
        key=lambda jp: jp.note_performance,
        reverse=True
    )[:6]
    for rang, jp in enumerate(classement, 1):
        print(f"  #{rang} Joueur {jp.joueur_id:3d} (Eq{jp.equipe_id})"
              f" | Note: {jp.note_performance:4.1f}"
              f" | Dist: {jp.distance_km:.2f}km"
              f" | Passes: {jp.passes_cles} clés"
              f" | Symétrie: {jp.symetrie_course:.0f}%"
              f" | Foulée: {jp.longueur_foulee_moy_m:.2f}m")

    return rapport


# ─────────────────────────────────────────────────────────────
# STEP 7 — Export JSON
# ─────────────────────────────────────────────────────────────

def exporter_json(rapport, chemin: str = "rapport_match.json"):
    print(f"\n[STEP 7] Export JSON → {chemin}")

    def _ser(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        if hasattr(obj, 'value'):
            return obj.value
        if isinstance(obj, (list, tuple)):
            return [_ser(i) for i in obj]
        return str(obj)

    data = {
        "version"         : "2.0",
        "duree_s"         : rapport.duree_s,
        "nb_frames"       : rapport.nb_frames_traitees,
        "evenements_total": rapport.evenements_total,
        "equipes": {
            str(eq_id): {
                "nom"                 : eq.nom,
                "possession_pct"      : eq.possession_pct,
                "tirs_total"          : eq.tirs_total,
                "tirs_cadres"         : eq.tirs_cadres,
                "expected_goals"      : eq.expected_goals,
                "buts"                : eq.buts,
                "passes_total"        : eq.passes_total,
                "passes_reussies"     : eq.passes_reussies,
                "taux_passes_pct"     : eq.taux_passes_pct,
                "passes_progressives" : eq.passes_progressives,
                "corners"             : eq.corners,
                "fautes"              : eq.fautes,
                "hors_jeux"           : eq.hors_jeux,
                "pressing_intensite"  : eq.pressing_intensite,
                "ppda"                : eq.ppda,
                "zones_recuperation"  : eq.zones_recuperation,
                "formation"           : eq.formation,
                "distance_totale_km"  : eq.distance_totale_km,
            }
            for eq_id, eq in rapport.equipes.items()
        },
        "joueurs": {
            str(jid): {
                "equipe_id"            : jp.equipe_id,
                "poste"                : jp.poste.value,
                "distance_km"          : jp.distance_km,
                "vitesse_max_kmh"      : jp.vitesse_max_kmh,
                "vitesse_moyenne_kmh"  : jp.vitesse_moyenne_kmh,
                "passes_total"         : jp.passes_total,
                "passes_reussies"      : jp.passes_reussies,
                "passes_cles"          : jp.passes_cles,
                "taux_passes_pct"      : jp.taux_passes_pct,
                "tirs_total"           : jp.tirs_total,
                "tirs_cadres"          : jp.tirs_cadres,
                "buts"                 : jp.buts,
                "dribbles_tentes"      : jp.dribbles_tentes,
                "dribbles_reussis"     : jp.dribbles_reussis,
                "duels_total"          : jp.duels_total,
                "duels_gagnes"         : jp.duels_gagnes,
                "taux_duels_pct"       : jp.taux_duels_pct,
                "interceptions"        : jp.interceptions,
                "tacles"               : jp.tacles,
                "tacles_reussis"       : jp.tacles_reussis,
                "position_moyenne"     : jp.position_moyenne,
                "zones_frequentees"    : jp.zones_frequentees,
                "note_performance"     : jp.note_performance,
                # Biomécanique v2
                "longueur_foulee_moy_m": jp.longueur_foulee_moy_m,
                "symetrie_course"      : jp.symetrie_course,
                "angle_inclination_moy": jp.angle_inclination_moy,
                "charge_genou"         : jp.charge_genou,
                "charge_hanche"        : jp.charge_hanche,
                # heatmap disponible via /joueurs/{id}/heatmap
            }
            for jid, jp in rapport.joueurs.items()
        }
    }

    with open(chemin, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  Export terminé : {chemin}")


# ─────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ANALYSE VIDÉO FOOTBALL  v2.0")
    print("  YOLOv9c | ByteTrack | ResNet-18 | YOLOv8-Pose")
    print("=" * 60)

    services = creer_services()
    (s1, s2, s3, s3b, s4, s5, s6, s7, s8, s9) = services

    initialiser_services(services)

    try:
        metadata = charger_video(s1)
    except FileNotFoundError as e:
        print(f"\nERREUR : {e}")
        print("Modifier CHEMIN_VIDEO dans main.py")
        sys.exit(1)

    calibrer_terrain(s1, s2)

    boucle_frames(services, fps=metadata.fps)

    calculer_stats_physiques(s4, s7, s9)

    rapport = generer_rapport(s9)

    exporter_json(rapport)

    s1.liberer()
    print("\nAnalyse terminée.")


if __name__ == "__main__":
    main()

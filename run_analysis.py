#!/usr/bin/env python
# run_analysis.py  — v2.0
#
# CLI wrapper pour le pipeline v2 (YOLOv9c + ByteTrack + ResNet-18 + YOLOv8-Pose)
# Lancé par l'API web (api_service.py) via subprocess
#
# Usage :
#   python run_analysis.py --video <chemin> [options]
#
# Options :
#   --equipe0        NAME   Nom équipe 0           (défaut: Equipe A)
#   --equipe1        NAME   Nom équipe 1           (défaut: Equipe B)
#   --output         PATH   Fichier JSON de sortie (défaut: rapport_match.json)
#   --max-frames     N      Nb max de frames (0 = toutes)
#   --modele         PATH   Modèle détection       (défaut: yolov9c.pt)
#   --modele-pose    PATH   Modèle pose            (défaut: yolov8n-pose.pt)
#   --progress-file  P      Fichier de progression (défaut: analyse_progress.json)

import argparse
import json
import sys
import logging
import traceback
from pathlib import Path

from config import settings

logging.basicConfig(level=logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
settings.configure_runtime_environment()


# ─────────────────────────────────────────────────────────────
# PROGRESSION
# ─────────────────────────────────────────────────────────────

def _write_progress(path: Path, etat: str, pct: int, message: str) -> None:
    data = {
        "etat":    etat,
        "pct":     max(0, min(100, pct)),
        "message": message,
    }
    try:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Progress write failed: {e}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────
# EXPORT JSON
# ─────────────────────────────────────────────────────────────

def _exporter_json(rapport, chemin: Path) -> None:
    def ser(obj):
        if hasattr(obj, '__dict__'):
            return {k: ser(v) for k, v in obj.__dict__.items()}
        if hasattr(obj, 'value'):
            return obj.value
        if isinstance(obj, (list, tuple)):
            return [ser(i) for i in obj]
        return str(obj)

    data = {
        "version":          "2.0",
        "duree_s":          rapport.duree_s,
        "nb_frames":        rapport.nb_frames_traitees,
        "evenements_total": rapport.evenements_total,
        "equipes": {
            str(eq_id): {
                "nom":                 eq.nom,
                "possession_pct":      round(float(eq.possession_pct or 0), 1),
                "tirs_total":          int(eq.tirs_total or 0),
                "tirs_cadres":         int(eq.tirs_cadres or 0),
                "expected_goals":      round(float(eq.expected_goals or 0), 2),
                "buts":                int(eq.buts or 0),
                "passes_total":        int(eq.passes_total or 0),
                "passes_reussies":     int(eq.passes_reussies or 0),
                "taux_passes_pct":     round(float(eq.taux_passes_pct or 0), 1),
                "passes_progressives": int(getattr(eq, 'passes_progressives', 0) or 0),
                "corners":             int(eq.corners or 0),
                "fautes":              int(eq.fautes or 0),
                "hors_jeux":           int(eq.hors_jeux or 0),
                "pressing_intensite":  round(float(eq.pressing_intensite or 0), 3),
                "ppda":                round(float(eq.ppda or 0), 2),
                "formation":           eq.formation or "N/A",
                "distance_totale_km":  round(float(eq.distance_totale_km or 0), 2),
                "zones_recuperation":  getattr(eq, 'zones_recuperation', {}),
            }
            for eq_id, eq in rapport.equipes.items()
        },
        "joueurs": {
            str(jid): {
                "equipe_id":             jp.equipe_id,
                "poste":                 jp.poste.value if hasattr(jp.poste, 'value') else str(jp.poste),
                "distance_km":           round(float(jp.distance_km or 0), 2),
                "vitesse_max_kmh":       round(float(jp.vitesse_max_kmh or 0), 1),
                "vitesse_moyenne_kmh":   round(float(jp.vitesse_moyenne_kmh or 0), 1),
                "passes_total":          int(jp.passes_total or 0),
                "passes_reussies":       int(jp.passes_reussies or 0),
                "passes_cles":           int(jp.passes_cles or 0),
                "taux_passes_pct":       round(float(jp.taux_passes_pct or 0), 1),
                "tirs_total":            int(jp.tirs_total or 0),
                "tirs_cadres":           int(jp.tirs_cadres or 0),
                "buts":                  int(jp.buts or 0),
                "dribbles_tentes":       int(jp.dribbles_tentes or 0),
                "dribbles_reussis":      int(jp.dribbles_reussis or 0),
                "duels_total":           int(jp.duels_total or 0),
                "duels_gagnes":          int(jp.duels_gagnes or 0),
                "taux_duels_pct":        round(float(jp.taux_duels_pct or 0), 1),
                "interceptions":         int(jp.interceptions or 0),
                "tacles":                int(jp.tacles or 0),
                "tacles_reussis":        int(jp.tacles_reussis or 0),
                "position_moyenne":      list(jp.position_moyenne),
                "zones_frequentees":     list(jp.zones_frequentees or []),
                "note_performance":      round(float(jp.note_performance or 0), 1),
                "heatmap":               getattr(jp, 'heatmap', []) or [],
                # Biomécanique v2
                "longueur_foulee_moy_m": round(float(getattr(jp, 'longueur_foulee_moy_m', 0) or 0), 2),
                "symetrie_course":       round(float(getattr(jp, 'symetrie_course', 100) or 100), 1),
                "angle_inclination_moy": round(float(getattr(jp, 'angle_inclination_moy', 0) or 0), 1),
                "charge_genou":          round(float(getattr(jp, 'charge_genou', 0) or 0), 1),
                "charge_hanche":         round(float(getattr(jp, 'charge_hanche', 0) or 0), 1),
            }
            for jid, jp in rapport.joueurs.items()
        },
    }

    with open(chemin, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def run(args, prog):
    """Pipeline v2 complet avec suivi de progression."""

    # ── Imports ────────────────────────────────────────────────
    prog("processing", 2, "Chargement des modules Python...")

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

    # ── Instanciation ──────────────────────────────────────────
    prog("processing", 4, "Instanciation des services v2...")

    s1  = VideoIngestionService()
    s2  = TerrainDetectionService()
    s3  = PlayerDetectionService(model_path=args.modele)
    s3b = PoseEstimationService(model_path=args.modele_pose)
    s4  = TrackingService()
    s5  = TeamClassificationService()
    s6  = BallTrackingService()
    s7  = PhysicalStatsService()
    s8  = TacticalAnalysisService()
    s9  = MatchAnalysisService(nom_equipe_0=args.equipe0, nom_equipe_1=args.equipe1)

    # ── Initialisation ─────────────────────────────────────────
    prog("processing", 5, "Initialisation (chargement modèles YOLOv9c + YOLOv8-Pose + ResNet-18)...")
    for svc in [s1, s2, s3, s3b, s4, s5, s6, s7, s8, s9]:
        svc.initialiser()

    # ── Chargement vidéo ───────────────────────────────────────
    prog("processing", 8, f"Chargement vidéo : {Path(args.video).name}")
    meta = s1.charger_video(args.video)
    total_frames = min(args.max_frames, meta.total_frames) if args.max_frames > 0 else meta.total_frames

    prog("processing", 10,
         f"Vidéo : {meta.largeur}×{meta.hauteur}, {total_frames} frames, "
         f"{meta.fps:.1f} fps, {meta.duree_minutes:.1f} min")

    # ── Calibration terrain ────────────────────────────────────
    prog("processing", 11, "Calibration terrain (homographie)...")
    frame0 = s1.extraire_frame(0)
    if frame0 is not None:
        s2.analyser_terrain(frame0)
        print(f"[INFO] Échelle terrain : {s2.pixels_par_metre_x:.1f} px/m", flush=True)

    # ── Configuration ──────────────────────────────────────────
    s4.fps              = int(meta.fps)
    s6.fps              = meta.fps
    s7.fps              = meta.fps
    s8.fps              = meta.fps
    s9.fps              = meta.fps
    s6.pixels_par_metre = s2.pixels_par_metre_x
    s7.pixels_par_metre = s2.pixels_par_metre_x
    s8.pixels_par_metre = s2.pixels_par_metre_x
    s9.pixels_par_metre = s2.pixels_par_metre_x
    s5.definir_noms_equipes(args.equipe0, args.equipe1)

    nb_frames  = 0
    PROG_START = 12
    PROG_END   = 85

    # ── Boucle frame par frame ─────────────────────────────────
    for frame, frame_id, timestamp in s1.generer_frames():
        if args.max_frames > 0 and nb_frames >= args.max_frames:
            break

        # S3 : YOLOv9c — joueurs + arbitres + ballon
        res_det     = s3.detecter(frame, frame_id, timestamp)
        joueurs_br  = res_det['joueurs']
        arbitres_br = res_det['arbitres']
        ballon_br   = res_det['ballon']

        # S4 : ByteTrack — IDs persistants
        joueurs_sv  = s4.mettre_a_jour(frame, joueurs_br, timestamp, frame_id)
        s4.mettre_a_jour_arbitres(arbitres_br, timestamp, frame_id)

        # S3b : YOLOv8-Pose — keypoints
        poses = s3b.estimer_poses(frame, joueurs_sv, frame_id)
        s7.accumuler_poses(poses)

        # S5 : ResNet-18 — classification équipes
        joueurs_cl = s5.classifier_tous(frame, joueurs_sv)
        for j in joueurs_cl:
            s4.assigner_equipe(j.id, j.equipe_id)

        # S6 : Ballon + possession
        rb     = s6.mettre_a_jour(ballon_br, joueurs_cl, timestamp, frame_id)
        ballon = rb.get('ballon')

        # S8 : Tactique (passes, pressing Voronoi, xG, formation)
        rt         = s8.analyser_frame(joueurs_cl, ballon, frame_id, timestamp)
        pressing   = rt.get('pressing', {})
        formation  = rt.get('formation', {})
        evts_frame = rt.get('evenements', [])

        # S9 : Accumulation
        s9.traiter({
            'joueurs':        joueurs_cl,
            'ballon':         ballon,
            'evenements':     evts_frame,
            'frame_id':       frame_id,
            'timestamp':      timestamp,
            'pressing_frame': pressing,
            'formation':      formation,
        })

        nb_frames += 1

        if nb_frames % 50 == 0:
            pct = int(PROG_START + (nb_frames / max(total_frames, 1)) * (PROG_END - PROG_START))
            prog("processing", min(pct, PROG_END),
                 f"Frame {frame_id} / ~{total_frames} "
                 f"({nb_frames * 100 // max(total_frames, 1)}%)"
                 f"  |  joueurs={len(joueurs_cl)}")

    prog("processing", 86, f"Boucle terminée : {nb_frames} frames.")

    # ── Stats physiques + biomécanique ─────────────────────────
    prog("processing", 88, "Calcul des stats physiques & biomécaniques...")
    stats_phys = s7.calculer_stats_tous_joueurs(s4.historique_positions)
    s9.integrer_stats_physiques(stats_phys)

    stats_bio = s7.calculer_biomecanique_tous()
    s9.integrer_stats_biomecaniques(stats_bio)

    # ── Rapport final ──────────────────────────────────────────
    prog("processing", 92, "Génération du rapport final...")
    rapport = s9.generer_rapport()

    # ── Export JSON ────────────────────────────────────────────
    output_path = Path(args.output)
    prog("processing", 96, f"Export JSON → {output_path.name}")
    _exporter_json(rapport, output_path)

    s1.liberer()

    n_joueurs = len(rapport.joueurs)
    prog("termine", 100,
         f"Analyse terminée. {n_joueurs} joueurs, 2 équipes. → {output_path.name}")
    print(f"[OK] Rapport généré : {output_path}  ({n_joueurs} joueurs)", flush=True)


# ─────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline v2 d'analyse vidéo football"
    )
    parser.add_argument("--video",        required=True,
                        help="Chemin vers la vidéo")
    parser.add_argument("--equipe0",      default="Equipe A")
    parser.add_argument("--equipe1",      default="Equipe B")
    parser.add_argument("--output",       default="rapport_match.json")
    parser.add_argument("--max-frames",   type=int, default=0,
                        help="Nb max frames (0 = toutes)")
    parser.add_argument("--modele",       default="yolov9c.pt",
                        help="Modèle détection YOLOv9c")
    parser.add_argument("--modele-pose",  default="yolov8n-pose.pt",
                        help="Modèle pose YOLOv8-Pose")
    parser.add_argument("--progress-file", default="analyse_progress.json")

    args = parser.parse_args()
    progress_path = Path(args.progress_file)

    def prog(etat: str, pct: int, message: str):
        _write_progress(progress_path, etat, pct, message)
        print(f"[{pct:3d}%] {message}", flush=True)

    try:
        run(args, prog)
    except Exception as e:
        detail = traceback.format_exc()
        _write_progress(progress_path, "erreur", 0, str(e))
        print(f"\n[ERREUR] {e}\n{detail}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

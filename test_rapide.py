# test_rapide.py
# Test du pipeline sur les 300 premieres frames uniquement
import sys, json
sys.stdout.reconfigure(line_buffering=True)

from core.data_models import Evenement, TypeEvenement, Position
from services.s01_ingestion.video_ingestion_service   import VideoIngestionService
from services.s02_terrain.terrain_detection_service   import TerrainDetectionService
from services.s03_detection.player_detection_service  import PlayerDetectionService
from services.s03b_pose.pose_estimation_service       import PoseEstimationService
from services.s04_tracking.tracking_service           import TrackingService
from services.s05_equipes.team_classification_service import TeamClassificationService
from services.s06_ballon.ball_tracking_service        import BallTrackingService
from services.s07_physique.physical_stats_service     import PhysicalStatsService
from services.s08_tactique.tactical_analysis_service  import TacticalAnalysisService
from services.s09_analyse_match.match_analysis_service import MatchAnalysisService
import logging
logging.basicConfig(level=logging.WARNING)   # Silencier les logs verbeux

CHEMIN_VIDEO = "video/match_wfc_Sonacos.mp4"
MAX_FRAMES   = 300   # Limiter pour le test

print("=" * 55)
print("  TEST RAPIDE — 300 frames")
print("=" * 55)

# Services
s1 = VideoIngestionService()
s2 = TerrainDetectionService()
s3 = PlayerDetectionService(model_path="yolov8n.pt")
s3b = PoseEstimationService(model_path="yolov8n-pose.pt")
s4 = TrackingService()
s5 = TeamClassificationService()
s6 = BallTrackingService()
s7 = PhysicalStatsService()
s8 = TacticalAnalysisService()
s9 = MatchAnalysisService(nom_equipe_0="WFC", nom_equipe_1="SONACOS")

for svc in [s1,s2,s3,s3b,s4,s5,s6,s7,s8,s9]:
    svc.initialiser()
print("[OK] Services initialises")

# Charger video
metadata = s1.charger_video(CHEMIN_VIDEO)
print(f"[OK] Video chargee : {metadata.total_frames} frames, {metadata.fps:.1f} FPS")

# Calibrer terrain
frame0 = s1.extraire_frame(0)
s2.analyser_terrain(frame0)
print(f"[OK] Terrain calibre : echelle={s2.pixels_par_metre_x:.2f} px/m")

# Propager echelle
fps = metadata.fps
s6.fps = fps; s7.fps = fps; s8.fps = fps; s9.fps = fps
s6.pixels_par_metre = s2.pixels_par_metre_x
s7.pixels_par_metre = s2.pixels_par_metre_x
s8.pixels_par_metre = s2.pixels_par_metre_x
s9.pixels_par_metre = s2.pixels_par_metre_x

# Boucle frames
_derniere = {}
evenements = []
nb = 0

for frame, frame_id, ts in s1.generer_frames():
    if nb >= MAX_FRAMES:
        break

    res3 = s3.detecter(frame, frame_id, ts)
    suivis = s4.mettre_a_jour(frame, res3['joueurs'], ts, frame_id)
    poses = s3b.estimer_poses(frame, suivis, frame_id)
    s7.accumuler_poses(poses)
    classes = s5.traiter({'frame': frame, 'joueurs': suivis})
    for j in classes:
        s4.assigner_equipe(j.id, j.equipe_id)
    res6 = s6.mettre_a_jour(res3['ballon'], classes, ts, frame_id)
    ballon = res6.get('ballon')
    res8 = s8.analyser_frame(classes, ballon, frame_id, ts)
    s9.traiter({
        'joueurs': classes, 'ballon': ballon,
        'evenements': res8.get('evenements', []), 'frame_id': frame_id,
        'timestamp': ts,
        'pressing_frame': res8.get('pressing', {}),
        'formation': res8.get('formation', {})
    })
    nb += 1
    if nb % 50 == 0:
        print(f"  frame {frame_id:4d} | joueurs={len(classes)} | ballon={'oui' if ballon else 'non'}")

print(f"\n[OK] {nb} frames traitees sans erreur")

# Stats physiques
hist = s4.historique_positions
stats_phys = s7.calculer_stats_tous_joueurs(hist)
s9.integrer_stats_physiques(stats_phys)
print(f"[OK] Stats physiques : {len(stats_phys)} joueurs")

# Rapport
rapport = s9.generer_rapport()
print(f"[OK] Rapport genere : {len(rapport.joueurs)} joueurs, {rapport.evenements_total} evenements")

for eq_id in [0,1]:
    eq = rapport.equipes[eq_id]
    print(f"  Equipe {eq_id} ({eq.nom}) : possession={eq.possession_pct}% | formation={eq.formation or 'N/A'}")

# Export JSON
with open("test_rapport.json","w",encoding="utf-8") as f:
    json.dump({
        "frames_traitees": nb,
        "joueurs": len(rapport.joueurs),
        "equipes": {str(k): {"nom": v.nom, "possession_pct": v.possession_pct}
                    for k,v in rapport.equipes.items()}
    }, f, ensure_ascii=False, indent=2)

print("[OK] Export JSON -> test_rapport.json")
print("\nTEST REUSSI")

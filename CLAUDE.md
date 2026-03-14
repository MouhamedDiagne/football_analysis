# CLAUDE.md — Football Analysis Project

Guide de référence pour Claude Code sur ce projet.

---

## Environnement

| Élément | Valeur |
|---|---|
| Python | `C:\Python313\python.exe` |
| Packages utilisateur | `C:\Users\algap\AppData\Roaming\Python\Python313\site-packages` |
| uvicorn | `C:\Users\algap\AppData\Roaming\Python\Python313\Scripts\uvicorn.exe` |
| OS | Windows 11, shell bash (Git Bash) |
| GPU | CPU uniquement (pas de CUDA) |

**Important** : `sys.executable` dans le serveur uvicorn pointe vers `C:\Python313\python.exe`.
Tous les packages sont installés au niveau utilisateur (`pip install --user`) ou dans `C:\Python313`.

---

## Architecture du pipeline (S1 → S12)  — v2.0

```
S1   VideoIngestionService      → Chargement vidéo, extraction frames
S2   TerrainDetectionService    → Homographie, calibration pixels/mètres
S3   PlayerDetectionService     → YOLOv9c : joueurs + ballon + arbitres
S3b  PoseEstimationService      → YOLOv8-Pose : 17 keypoints COCO par joueur
S4   TrackingService            → ByteTrack (supervision) : ID persistant
S5   TeamClassificationService  → ResNet-18 features + KMeans → équipe 0/1
S6   BallTrackingService        → Tracking ballon, possession, équipe
S7   PhysicalStatsService       → Distance, vitesse, biomécanique (foulée, symétrie, angles)
S8   TacticalAnalysisService    → Passes, pressing Voronoi, xG, heatmaps 12x12, formations
S9   MatchAnalysisService       → Accumulation stats match, rapport final
S10  StatsGeneratorService      → (stub .txt, pas encore implémenté)
S11  ReportGeneratorService     → (stub .txt, pas encore implémenté)
S12  api_service.py             → FastAPI web + REST sur port 8000
```

**Calibration critique** : après `s2.analyser_terrain()`, toujours propager :
```python
s7.pixels_par_metre = s2.pixels_par_metre_x
s8.pixels_par_metre = s2.pixels_par_metre_x
```
Sans ça, distances et pressing sont faux (calcul en pixels au lieu de mètres).

---

## Bugs connus et correctifs appliqués

### 1. Seuil pressing en pixels (pas en mètres)
**Fichier** : `services/s08_tactique/tactical_analysis_service.py`
```python
# CORRECT : convertir 5m en pixels
seuil_pixels = 5.0 * self.pixels_par_metre
if dist < seuil_pixels:
    pressing[eq_id] += 1.0 / (dist + 0.1)
```

### 2. Ballon.bbox doit être Optional
**Fichier** : `core/data_models.py`
```python
@dataclass
class Ballon:
    position: Position
    bbox: Optional[BoundingBox]  # était BoundingBox (non-optional → crash)
    confiance: float
```

### 3. Formation : guard à frame 0
**Fichier** : `services/s08_tactique/tactical_analysis_service.py`
```python
# Vérifier que la liste n'est pas vide avant modulo
if len(self.positions_equipe[0]) > 0 and len(self.positions_equipe[0]) % 50 == 0:
```

### 4. Dead code dans _evt_faute
**Fichier** : `services/s09_analyse_match/match_analysis_service.py`
```python
# SUPPRIMÉ : jp.tacles += 0  (incrémentait de zéro = inutile)
```

### 5. float32 → float64 pour sklearn 1.8
**Fichier** : `services/s05_equipes/team_classification_service.py`
```python
pixels = zone_rgb.reshape(-1, 3).astype(np.float64)  # était float32
```

### 6. pkg_resources incompatible Python 3.13
**Fichier** : `C:\Users\algap\AppData\Roaming\Python\Python313\site-packages\deep_sort_realtime\embedder\embedder_pytorch.py`
```python
# Remplacer pkg_resources par os.path :
import os
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOBILENETV2_BOTTLENECK_WTS = os.path.join(_PACKAGE_DIR, "embedder", "weights", "mobilenetv2_bottleneck_wts.pt")
```

---

## Fichiers clés

| Fichier | Rôle |
|---|---|
| `main.py` | Runner pipeline original (hardcodé) |
| `run_analysis.py` | Runner CLI avec suivi progression (pour l'API web) |
| `test_rapide.py` | Test 300 frames (~2 min CPU) |
| `rapport_match.json` | Rapport généré (sortie standard) |
| `analyse_progress.json` | État en temps réel de l'analyse (polling) |
| `video/match_wfc_Sonacos.mp4` | Vidéo de référence |
| `video/uploads/` | Vidéos uploadées via l'interface web |
| `services/s12_api/index.html` | Interface web complète |
| `services/s12_api/api_service.py` | Serveur FastAPI v2.0 |
| `.claude/launch.json` | Config serveurs de dev |

---

## Interface Web (S12)

**URL** : `http://localhost:8000`

| Endpoint | Méthode | Description |
|---|---|---|
| `/` | GET | Interface HTML (upload + rapport) |
| `/upload` | POST | Upload vidéo + lancement analyse subprocess |
| `/status` | GET | État de l'analyse (polling toutes les 2s) |
| `/rapport` | GET | Rapport JSON complet |
| `/download/rapport` | GET | Téléchargement JSON |
| `/equipes` | GET | Stats des 2 équipes |
| `/joueurs` | GET | Liste joueurs (filtrable, triable) |
| `/joueurs/{id}/heatmap` | GET | Heatmap 12×12 |
| `/comparaison` | GET | Comparaison côte-à-côte |
| `/classement` | GET | Classement joueurs |
| `/health` | GET | Santé du serveur |
| `/docs` | GET | Swagger UI automatique |

**Lancer le serveur** :
```bash
"/c/Users/algap/AppData/Roaming/Python/Python313/Scripts/uvicorn.exe" \
  services.s12_api.api_service:app \
  --host 0.0.0.0 --port 8000 --reload
```

---

## Performance CPU

| Max frames | Temps traitement | Couverture vidéo | Usage recommandé |
|---|---|---|---|
| `300` | ~2 min | ~12 s | Test / débogage |
| `1 000` | ~7 min | ~40 s | Démo |
| **`3 000`** | **~20 min** | **~2 min** | **Optimal CPU** |
| `7 500` | ~50 min | ~5 min | Analyse détaillée |
| `0` (toutes) | ~15-20 h | 90 min | GPU uniquement |

**Référence** : vitesse constatée ≈ 2–3 frames/seconde sur CPU (YOLOv8n + DeepSORT + KMeans).

---

## Structure rapport JSON (rapport_match.json)

```json
{
  "duree_s": float,
  "nb_frames": int,
  "evenements_total": int,
  "equipes": {
    "0": { "nom": str, "possession_pct": float, "tirs_total": int,
           "distance_totale_km": float, "formation": str, ... },
    "1": { ... }
  },
  "joueurs": {
    "2": { "equipe_id": int, "distance_km": float, "vitesse_max_kmh": float,
           "note_performance": float, "heatmap": [[12x12 grid 0-100]], ... },
    ...
  }
}
```

---

## Dépendances (requirements.txt)  — v2.0

```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.2.0        # YOLOv9c + YOLOv8-Pose
supervision>=0.21.0       # ByteTrack (remplace deep-sort-realtime)
torch>=2.0.0              # ResNet-18
torchvision>=0.15.0       # ResNet-18
scikit-learn>=1.3.0
scipy>=1.11.0
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
```

**Installation** :
```bash
pip install --user ultralytics supervision torch torchvision scikit-learn scipy fastapi uvicorn python-multipart opencv-python numpy
```

---

## Conventions de développement

- Les services héritent de `BaseService` (`core/base_service.py`)
- Chaque service implémente `initialiser()` et sa méthode principale
- Les données circulent en `dict` entre services (pas d'objets fortement typés entre stages)
- `rapport_match.json` est la sortie standard — toujours écrire dans ce fichier
- `analyse_progress.json` suit le schéma `{"etat": str, "pct": int, "message": str}`
- Les heatmaps sont des listes de 12 listes de 12 float (0–100), ligne = axe Y

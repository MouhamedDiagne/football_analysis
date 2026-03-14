# services/s12_api/api_service.py
#
# Football Analysis — Web Interface + REST API  (v2.0)
#
# Endpoints principaux :
#   GET  /                        → Interface web (HTML)
#   POST /upload                  → Upload video + lancement analyse
#   GET  /status                  → Etat de l'analyse (polling frontend)
#   GET  /rapport                 → Rapport complet (JSON)
#   GET  /download/rapport        → Telechargement du JSON
#   GET  /equipes                 → Stats des deux equipes
#   GET  /equipes/{id}            → Stats d'une equipe
#   GET  /joueurs                 → Liste joueurs (filtrable, triable)
#   GET  /joueurs/{id}            → Stats d'un joueur
#   GET  /joueurs/{id}/heatmap    → Heatmap 12x12
#   GET  /comparaison             → Comparaison cote-a-cote
#   GET  /classement              → Classement joueurs
#   POST /rapport/recharger       → Vider le cache JSON
#   GET  /health                  → Etat serveur

import json
import sys
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

from config import settings
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

settings.configure_runtime_environment()

# ─── Chemins ──────────────────────────────────────────────────
BASE_DIR      = settings.BASE_DIR
RAPPORT_JSON  = settings.REPORT_PATH
PROGRESS_FILE = settings.PROGRESS_PATH
UPLOAD_DIR    = settings.UPLOAD_DIR
INDEX_HTML    = settings.INDEX_HTML_PATH

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─── Application ──────────────────────────────────────────────
app = FastAPI(
    title="Football Analysis API",
    description=(
        "Interface Web + API REST pour l'analyse video de match de football. "
        "Visitez / pour l'interface graphique."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Cache & etat ─────────────────────────────────────────────
_rapport_cache: Optional[dict] = None
_analyse_process: Optional[subprocess.Popen] = None


# ─────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────

def _charger_rapport() -> dict:
    global _rapport_cache
    if _rapport_cache is not None:
        return _rapport_cache
    if not RAPPORT_JSON.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Rapport non disponible. "
                "Lancez d'abord une analyse via l'interface web (GET /)."
            ),
        )
    with open(RAPPORT_JSON, encoding="utf-8") as f:
        _rapport_cache = json.load(f)
    return _rapport_cache


def _recharger_rapport() -> dict:
    global _rapport_cache
    _rapport_cache = None
    return _charger_rapport()


def _lire_progression() -> dict:
    """Lire le fichier de progression de l'analyse en cours."""
    if not PROGRESS_FILE.exists():
        return {"etat": "idle", "pct": 0, "message": "", "rapport_disponible": RAPPORT_JSON.exists()}
    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        data["rapport_disponible"] = RAPPORT_JSON.exists()
        return data
    except Exception:
        return {"etat": "idle", "pct": 0, "message": "", "rapport_disponible": RAPPORT_JSON.exists()}


def _analyse_active() -> bool:
    """True si un subprocess d'analyse est en train de tourner."""
    global _analyse_process
    if _analyse_process is None:
        return False
    return _analyse_process.poll() is None  # poll() == None → encore en vie


def _lire_index_html() -> str:
    """Lire l'interface HTML depuis le fichier index.html."""
    if INDEX_HTML.exists():
        return INDEX_HTML.read_text(encoding="utf-8")
    # Fallback minimaliste si le fichier est absent
    return """<!DOCTYPE html><html><body>
    <h2>Football Analysis API</h2>
    <p>Le fichier <code>services/s12_api/index.html</code> est introuvable.</p>
    <p><a href="/docs">Voir la documentation API →</a></p>
    </body></html>"""


# ─────────────────────────────────────────────────────────────
# ENDPOINTS — INTERFACE WEB
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Interface Web"])
def index():
    """Interface web principale — Upload vidéo, progression, rapport."""
    return HTMLResponse(content=_lire_index_html())


# ─────────────────────────────────────────────────────────────
# ENDPOINTS — ANALYSE
# ─────────────────────────────────────────────────────────────

@app.get("/status", tags=["Analyse"])
def get_status():
    """
    Etat courant de l'analyse.
    Retourne : { etat, pct, message, rapport_disponible }
    etat : 'idle' | 'processing' | 'termine' | 'erreur'
    """
    global _analyse_process, _rapport_cache

    prog = _lire_progression()
    prog["rapport_disponible"] = RAPPORT_JSON.exists()

    # Si un processus tourne, verifier sa fin
    if _analyse_process is not None:
        retcode = _analyse_process.poll()
        if retcode is not None:
            # Processus termine
            if retcode == 0 and prog.get("etat") not in ("termine", "erreur"):
                prog = {
                    "etat":    "termine",
                    "pct":     100,
                    "message": "Analyse terminee.",
                    "rapport_disponible": RAPPORT_JSON.exists(),
                }
            elif retcode != 0 and prog.get("etat") not in ("erreur",):
                prog = {
                    "etat":    "erreur",
                    "pct":     0,
                    "message": f"Processus termine avec code d'erreur {retcode}.",
                    "rapport_disponible": RAPPORT_JSON.exists(),
                }

    # Si l'analyse vient de se terminer, invalider le cache du rapport
    if prog.get("etat") == "termine":
        _rapport_cache = None

    return prog


@app.post("/upload", tags=["Analyse"])
async def upload_video(
    fichier:    UploadFile = File(...,  description="Fichier video a analyser"),
    equipe0:    str        = Form("Equipe A", description="Nom equipe 0"),
    equipe1:    str        = Form("Equipe B", description="Nom equipe 1"),
    max_frames: int        = Form(0,  description="Nb max de frames (0 = toutes)"),
):
    """
    Upload une vidéo et lance le pipeline d'analyse en arrière-plan.
    Retourne 202 Accepted immédiatement — suivre /status pour la progression.
    """
    global _analyse_process, _rapport_cache

    # ── Vérifier qu'une analyse n'est pas déjà en cours ───────
    if _analyse_active():
        raise HTTPException(
            status_code=409,
            detail="Une analyse est déjà en cours. Attendez qu'elle se termine.",
        )

    # ── Valider le type de fichier ─────────────────────────────
    if not fichier.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant.")

    ext = Path(fichier.filename).suffix.lower()
    EXTENSIONS_OK = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts"}
    if ext not in EXTENSIONS_OK:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : '{ext}'. Acceptés : {', '.join(sorted(EXTENSIONS_OK))}",
        )

    # ── Sauvegarder la vidéo ───────────────────────────────────
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = UPLOAD_DIR / f"match_{ts}{ext}"

    try:
        with open(video_path, "wb") as out:
            CHUNK = 1024 * 1024  # 1 MB
            while True:
                chunk = await fichier.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur sauvegarde vidéo : {e}")

    # ── Écrire la progression initiale ────────────────────────
    PROGRESS_FILE.write_text(
        json.dumps({"etat": "processing", "pct": 0, "message": "Démarrage de l'analyse..."},
                   ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Invalider le cache du rapport précédent ───────────────
    _rapport_cache = None

    # ── Construire la commande subprocess ─────────────────────
    python_exe = sys.executable       # Même Python que le serveur
    script     = BASE_DIR / "run_analysis.py"

    cmd = [
        python_exe, str(script),
        "--video",         str(video_path),
        "--equipe0",       equipe0.strip() or "Equipe A",
        "--equipe1",       equipe1.strip() or "Equipe B",
        "--output",        str(RAPPORT_JSON),
        "--progress-file", str(PROGRESS_FILE),
    ]
    if max_frames > 0:
        cmd += ["--max-frames", str(max_frames)]

    # ── Lancer le subprocess ──────────────────────────────────
    try:
        _analyse_process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),          # Repertoire racine du projet
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Sur Windows, ne pas heriter du terminal interactif
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except Exception as e:
        PROGRESS_FILE.write_text(
            json.dumps({"etat": "erreur", "pct": 0, "message": str(e)}, ensure_ascii=False),
            encoding="utf-8",
        )
        raise HTTPException(status_code=500, detail=f"Impossible de démarrer l'analyse : {e}")

    return JSONResponse(
        status_code=202,
        content={
            "status":     "accepted",
            "message":    f"Analyse démarrée pour '{fichier.filename}'",
            "video_path": str(video_path),
            "pid":        _analyse_process.pid,
        },
    )


# ─────────────────────────────────────────────────────────────
# ENDPOINTS — RAPPORT
# ─────────────────────────────────────────────────────────────

@app.get("/rapport", tags=["Rapport"])
def get_rapport():
    """Rapport complet du match au format JSON."""
    return _charger_rapport()


@app.get("/download/rapport", tags=["Rapport"])
def download_rapport():
    """Télécharger le fichier rapport_match.json."""
    if not RAPPORT_JSON.exists():
        raise HTTPException(status_code=404, detail="Rapport non disponible.")
    return FileResponse(
        path=str(RAPPORT_JSON),
        media_type="application/json",
        filename="rapport_match.json",
    )


@app.post("/rapport/recharger", tags=["Admin"])
def recharger_rapport():
    """Vider le cache et recharger le rapport depuis le fichier JSON."""
    try:
        rapport = _recharger_rapport()
        return {
            "status":  "ok",
            "message": "Rapport rechargé",
            "joueurs": len(rapport.get("joueurs", {})),
            "equipes": len(rapport.get("equipes", {})),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# ENDPOINTS — ÉQUIPES
# ─────────────────────────────────────────────────────────────

@app.get("/equipes", tags=["Équipes"])
def get_equipes():
    """Stats des deux équipes."""
    return _charger_rapport().get("equipes", {})


@app.get("/equipes/{equipe_id}", tags=["Équipes"])
def get_equipe(equipe_id: int):
    """Stats détaillées d'une équipe (0 ou 1)."""
    equipes = _charger_rapport().get("equipes", {})
    cle = str(equipe_id)
    if cle not in equipes:
        raise HTTPException(
            status_code=404,
            detail=f"Équipe {equipe_id} introuvable. IDs disponibles : {list(equipes.keys())}",
        )
    return equipes[cle]


# ─────────────────────────────────────────────────────────────
# ENDPOINTS — JOUEURS
# ─────────────────────────────────────────────────────────────

@app.get("/joueurs", tags=["Joueurs"])
def get_joueurs(
    equipe_id: Optional[int] = Query(None, description="Filtrer par equipe (0 ou 1)"),
    tri:       str            = Query("note_performance", description="Champ de tri"),
    ordre:     str            = Query("desc", description="asc ou desc"),
    limite:    int            = Query(100, ge=1, le=500, description="Nombre max"),
):
    """
    Liste tous les joueurs.
    Filtrable par équipe, triable par n'importe quel champ numérique.
    """
    joueurs   = _charger_rapport().get("joueurs", {})
    resultats = []
    for jid, stats in joueurs.items():
        if equipe_id is not None and stats.get("equipe_id") != equipe_id:
            continue
        resultats.append({"joueur_id": int(jid), **stats})

    reverse = ordre.lower() != "asc"
    try:
        resultats.sort(key=lambda j: j.get(tri) or 0, reverse=reverse)
    except Exception:
        pass

    return {"total": len(resultats), "joueurs": resultats[:limite]}


@app.get("/joueurs/{joueur_id}", tags=["Joueurs"])
def get_joueur(joueur_id: int):
    """Stats complètes d'un joueur."""
    joueurs = _charger_rapport().get("joueurs", {})
    cle = str(joueur_id)
    if cle not in joueurs:
        ids = sorted(int(k) for k in joueurs)
        raise HTTPException(
            status_code=404,
            detail=f"Joueur {joueur_id} introuvable. IDs disponibles : {ids}",
        )
    return {"joueur_id": joueur_id, **joueurs[cle]}


@app.get("/joueurs/{joueur_id}/heatmap", tags=["Joueurs"])
def get_heatmap(joueur_id: int):
    """
    Heatmap 12×12 d'occupation du terrain.
    Valeurs normalisées 0–100.
    Lignes = axe Y (haut → bas), Colonnes = axe X (gauche → droite).
    """
    joueurs = _charger_rapport().get("joueurs", {})
    cle = str(joueur_id)
    if cle not in joueurs:
        raise HTTPException(status_code=404, detail=f"Joueur {joueur_id} introuvable.")
    heatmap = joueurs[cle].get("heatmap", [])
    return {
        "joueur_id":  joueur_id,
        "grille":     "12x12",
        "description": "Lignes=axe Y (haut→bas), Colonnes=axe X (gauche→droite)",
        "heatmap":    heatmap,
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINTS — ANALYSE
# ─────────────────────────────────────────────────────────────

@app.get("/comparaison", tags=["Analyse"])
def get_comparaison():
    """Comparaison côte-à-côte des deux équipes."""
    equipes  = _charger_rapport().get("equipes", {})
    metriques = [
        "possession_pct", "tirs_total", "tirs_cadres", "expected_goals",
        "buts", "passes_total", "passes_reussies", "taux_passes_pct",
        "passes_progressives", "corners", "fautes", "hors_jeux",
        "pressing_intensite", "ppda", "formation", "distance_totale_km",
    ]
    return {
        "equipes": {
            eq_id: stats.get("nom", f"Équipe {eq_id}")
            for eq_id, stats in equipes.items()
        },
        "comparaison": {
            m: {eq_id: stats.get(m, 0) for eq_id, stats in equipes.items()}
            for m in metriques
        },
    }


@app.get("/classement", tags=["Analyse"])
def get_classement(
    top:       int          = Query(10, ge=1, le=50, description="Nombre de joueurs"),
    par:       str          = Query("note_performance", description="Critère de classement"),
    equipe_id: Optional[int] = Query(None, description="Filtrer par équipe"),
):
    """
    Classement des meilleurs joueurs.
    Critères : note_performance, distance_km, vitesse_max_kmh,
               passes_cles, tirs_total, buts, interceptions.
    """
    joueurs   = _charger_rapport().get("joueurs", {})
    resultats = [
        {"joueur_id": int(jid), **stats}
        for jid, stats in joueurs.items()
        if equipe_id is None or stats.get("equipe_id") == equipe_id
    ]
    resultats.sort(key=lambda j: j.get(par) or 0, reverse=True)
    return {
        "critere":   par,
        "classement": [{"rang": i + 1, **j} for i, j in enumerate(resultats[:top])],
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINTS — INFO
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Info"])
def health():
    """Etat du serveur."""
    return {
        "status":         "ok",
        "rapport_pret":   RAPPORT_JSON.exists(),
        "analyse_active": _analyse_active(),
        "rapport_path":   str(RAPPORT_JSON),
        "runtime":        settings.runtime_summary(),
    }


# ─────────────────────────────────────────────────────────────
# POINT D'ENTREE DIRECT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  FOOTBALL ANALYSIS PLATFORM  v2.0")
    print("  Interface : http://localhost:8000")
    print("  API Docs  : http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(
        "services.s12_api.api_service:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.UVICORN_RELOAD,
        log_level="info",
    )

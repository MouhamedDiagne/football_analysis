"""
fix_yolov9c.py
Télécharge yolov9c.pt directement depuis GitHub releases
et écrase le fichier corrompu existant.
"""
import sys
import pathlib
import urllib.request
import shutil
import os

URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9c.pt"
DEST = pathlib.Path(__file__).parent / "yolov9c.pt"
TEMP = pathlib.Path(__file__).parent / "yolov9c_dl_temp.pt"

TAILLE_ATTENDUE_MB = 100   # yolov9c fait ~100-104 MB

def reporthook(count, block_size, total_size):
    pct = int(count * block_size * 100 / max(total_size, 1))
    mb_done = count * block_size / 1e6
    mb_total = total_size / 1e6
    print(f"\r  {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)

print(f"Téléchargement yolov9c.pt depuis :")
print(f"  {URL}")
print(f"Destination temporaire : {TEMP}")
print()

try:
    # Supprimer le fichier temp s'il existe déjà
    if TEMP.exists():
        TEMP.unlink()

    urllib.request.urlretrieve(URL, TEMP, reporthook)
    print()  # saut de ligne après la barre

    size_mb = TEMP.stat().st_size / 1e6
    print(f"\nTéléchargé : {size_mb:.1f} MB")

    if size_mb < TAILLE_ATTENDUE_MB * 0.5:
        print(f"[ERREUR] Taille trop petite ({size_mb:.1f} MB < {TAILLE_ATTENDUE_MB} MB attendus)")
        sys.exit(1)

    # Remplacer le fichier de destination
    if DEST.exists():
        # Essayer de supprimer l'ancien
        try:
            DEST.unlink()
            print(f"Ancien fichier supprimé : {DEST}")
        except PermissionError:
            # Renommer l'ancien si verrouillé
            backup = DEST.with_suffix(".pt.old")
            try:
                DEST.rename(backup)
                print(f"Ancien fichier renommé en : {backup.name}")
            except Exception as e2:
                print(f"[WARN] Impossible de déplacer l'ancien fichier : {e2}")
                print(f"  → Le nouveau fichier est disponible sous : {TEMP.name}")
                print(f"  → Renommez manuellement : {TEMP.name} → yolov9c.pt")
                sys.exit(0)

    # Déplacer le temp vers la destination
    TEMP.rename(DEST)
    print(f"\n[OK] yolov9c.pt installé ({DEST.stat().st_size / 1e6:.1f} MB) → {DEST}")

except Exception as e:
    print(f"\n[ERREUR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

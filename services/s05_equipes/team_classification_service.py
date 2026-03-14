# services/s05_equipes/team_classification_service.py  — v2.0
# Classification d'équipe via ResNet-18 feature extraction + KMeans

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
from config.settings import get_model_device
from core.base_service import BaseService
from core.data_models import Joueur


class TeamClassificationService(BaseService):
    """
    Classification d'équipe (0 ou 1) par couleur maillot.

    Méthode v2 — ResNet-18 Feature Extraction :
    1. Crop le patch torse du joueur
    2. Extrait un vecteur 512-dim via ResNet-18 pré-entraîné (sans couche FC)
    3. Phase d'entraînement : accumule les features pendant N frames
    4. Ajuste KMeans(k=2) sur ces features
    5. Classification : prédiction + vote majoritaire sur 15 frames

    Avantages sur KMeans couleur brute :
    - Robuste aux variations d'éclairage et d'ombre
    - Capte textures + couleurs (numéro, bandes, etc.)
    - Plus stable sur des maillots proches en couleur
    """

    def __init__(self, nb_frames_entrainement: int = 50):
        super().__init__("TeamClassification")
        self.nb_frames_entrainement = nb_frames_entrainement

        self.model     = None   # ResNet-18 tronqué
        self.transform = None   # Pipeline torchvision
        self.kmeans    = None   # KMeans(k=2)
        self.device    = get_model_device()

        self.noms_equipes: Dict[int, str] = {0: "Equipe A", 1: "Equipe B"}

        # Vote majoritaire par joueur (fenêtre glissante de 15 frames)
        self.votes_equipe: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=15)
        )

        # Données collectées pour l'entraînement
        self._features_entrainement: List[np.ndarray] = []
        self._frame_count     = 0
        self._est_entraine    = False

    # ─────────────────────────────────────
    # INITIALISATION
    # ─────────────────────────────────────

    def initialiser(self):
        import torch
        import torchvision.models   as models
        import torchvision.transforms as transforms

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Supprime la couche FC finale → sortie 512-dim (avgpool)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

        self.logger.info("✅ ResNet-18 feature extractor prêt (512-dim)")
        self.logger.info(f"   Device ResNet: {self.device}")
        self.est_initialise = True

    def traiter(self, data: dict) -> List[Joueur]:
        return self.classifier_tous(data['frame'], data['joueurs'])

    # ─────────────────────────────────────
    # EXTRACTION DE FEATURES
    # ─────────────────────────────────────

    def _extraire_feature(self,
                           frame: np.ndarray,
                           bbox) -> Optional[np.ndarray]:
        """
        Extrait un vecteur ResNet-18 de 512 dimensions
        depuis la zone torse du joueur.
        """
        import torch

        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        h  = y2 - y1
        ty1 = y1 + int(h * 0.20)
        ty2 = y1 + int(h * 0.55)

        if ty1 >= ty2 or x1 >= x2:
            return None

        crop = frame[ty1:ty2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor   = self.transform(crop_rgb).unsqueeze(0).to(self.device)   # (1, 3, 64, 32)

        with torch.no_grad():
            feat = self.model(tensor)   # (1, 512, 1, 1)

        return feat.squeeze().detach().cpu().numpy()   # (512,)

    # ─────────────────────────────────────
    # ENTRAÎNEMENT
    # ─────────────────────────────────────

    def entrainer(self, frame: np.ndarray, joueurs: List[Joueur]) -> bool:
        """
        Collecte des features sur les premières frames,
        puis entraîne KMeans(k=2).
        """
        from sklearn.cluster import KMeans

        for j in joueurs:
            feat = self._extraire_feature(frame, j.bbox)
            if feat is not None:
                self._features_entrainement.append(feat)

        self._frame_count += 1

        if (self._frame_count >= self.nb_frames_entrainement
                and not self._est_entraine
                and len(self._features_entrainement) >= 4):

            X = np.array(self._features_entrainement)
            self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            self.kmeans.fit(X)
            self._est_entraine = True

            self.logger.info(
                f"✅ KMeans entraîné sur {len(X)} features ResNet-18"
            )
            return True

        return False

    # ─────────────────────────────────────
    # CLASSIFICATION
    # ─────────────────────────────────────

    def classifier_joueur(self,
                           frame: np.ndarray,
                           joueur: Joueur) -> int:
        """Classifie un joueur → équipe 0 ou 1 (vote majoritaire)."""
        if not self._est_entraine or self.kmeans is None:
            return joueur.equipe_id if joueur.equipe_id >= 0 else 0

        feat = self._extraire_feature(frame, joueur.bbox)
        if feat is None:
            votes = list(self.votes_equipe[joueur.id])
            return max(set(votes), key=votes.count) if votes else 0

        prediction = int(self.kmeans.predict(feat.reshape(1, -1))[0])
        self.votes_equipe[joueur.id].append(prediction)

        votes = list(self.votes_equipe[joueur.id])
        return max(set(votes), key=votes.count)

    def classifier_tous(self,
                         frame: np.ndarray,
                         joueurs: List[Joueur]) -> List[Joueur]:
        """Classifie tous les joueurs de la frame."""
        if not self._est_entraine:
            self.entrainer(frame, joueurs)
            return joueurs

        for j in joueurs:
            j.equipe_id = self.classifier_joueur(frame, j)

        return joueurs

    # ─────────────────────────────────────
    # CONFIGURATION
    # ─────────────────────────────────────

    def definir_noms_equipes(self, nom_eq0: str, nom_eq1: str):
        self.noms_equipes = {0: nom_eq0, 1: nom_eq1}

    def get_nom_equipe(self, equipe_id: int) -> str:
        return self.noms_equipes.get(equipe_id, f"Equipe {equipe_id}")

    @property
    def est_entraine(self) -> bool:
        return self._est_entraine

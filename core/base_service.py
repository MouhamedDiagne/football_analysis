# core/base_service.py

from abc import ABC, abstractmethod
import logging
import time

class BaseService(ABC):
    """
    Classe de base pour tous les services
    """

    def __init__(self, nom_service: str):
        self.nom = nom_service
        self.logger = logging.getLogger(nom_service)
        self._configure_logger()
        self.temps_execution = 0
        self.erreurs = []
        self.est_initialise = False

    def _configure_logger(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'[%(asctime)s] [{self.nom}] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    @abstractmethod
    def initialiser(self):
        """Initialiser le service"""
        pass

    @abstractmethod
    def traiter(self, data):
        """Traiter les données"""
        pass

    def executer(self, data):
        """Exécuter le service avec mesure du temps"""
        debut = time.time()
        try:
            self.logger.info(f"▶️  Démarrage...")
            resultat = self.traiter(data)
            self.temps_execution = time.time() - debut
            self.logger.info(
                f"✅ Terminé en {self.temps_execution:.2f}s"
            )
            return resultat
        except Exception as e:
            self.erreurs.append(str(e))
            self.logger.error(f"❌ Erreur: {e}")
            raise

    def get_status(self) -> dict:
        return {
            "service": self.nom,
            "initialise": self.est_initialise,
            "temps_execution": self.temps_execution,
            "nb_erreurs": len(self.erreurs)
        }
# ==============================================================================
# UTILITAIRES DE GESTION DES VARIABLES D'ENVIRONNEMENT
# ==============================================================================

import os
from pathlib import Path


def detect_environment():
    """
    Détecte si l'application s'exécute en local, Docker ou sur Streamlit Cloud.
    
    Returns:
        str: "local", "docker" ou "cloud"
    """
    # Streamlit Cloud ajoute toujours cette variable d'environnement
    if os.getenv("STREAMLIT_SERVER_HEADLESS") == "true":
        return "cloud"
    # Vérifier si on est dans un conteneur Docker
    if os.path.exists("/.dockerenv"):
        return "docker"
    # Sinon, on est en local
    return "local"


def get_env_variable(var_name, default=""):
    """
    Récupère une variable d'environnement de manière sécurisée selon l'environnement.
    
    En production (Streamlit Cloud):
    - Utilise st.secrets en priorité
    - Fallback sur les variables d'environnement système
    
    En développement (local/Docker):
    - Utilise les variables d'environnement système
    
    Args:
        var_name (str): Nom de la variable d'environnement
        default (str): Valeur par défaut si non trouvée
        
    Returns:
        str: Valeur de la variable ou valeur par défaut
    """
    environment = detect_environment()
    
    # Essayer st.secrets pour Streamlit Cloud
    if environment == "cloud":
        try:
            import streamlit as st
            try:
                if hasattr(st, 'secrets') and var_name in st.secrets:
                    return st.secrets[var_name]
            except Exception:
                # st.secrets non disponible ou erreur d'accès
                pass
        except ImportError:
            # Streamlit n'est pas disponible
            pass
    
    # Fallback: variables d'environnement système
    env_value = os.getenv(var_name)
    if env_value is not None:
        return env_value
    
    # Valeur par défaut
    return default if default else ""


def get_auth_config_path():
    """
    Récupère la configuration d'authentification.
    
    En local/Docker:
    - Retourne le chemin vers config.yaml (fichier)
    
    En Streamlit Cloud:
    - Si config.yaml n'existe pas, récupère depuis st.secrets
    - Crée un fichier temp avec la config TOML depuis st.secrets
    
    Returns:
        Path: Chemin vers le fichier config.yaml
        
    Raises:
        FileNotFoundError: Si aucune config trouvée
    """
    import yaml
    import tempfile
    
    environment = detect_environment()
        
    # En Streamlit Cloud, chercher dans les secrets TOML
    if environment == "cloud":
        try:
            import streamlit as st
            
            # Vérifier si on a les credentials dans les secrets
            if hasattr(st, 'secrets') and 'credentials' in st.secrets:
                try:
                    # Récupérer la structure credentials depuis st.secrets (TOML parsé)
                    config_dict = {
                        'credentials': dict(st.secrets['credentials'])
                    }
                    
                    # Créer un fichier temporaire avec le contenu
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
                        yaml.dump(config_dict, f)
                        temp_path = Path(f.name)
                    
                    return temp_path
                except Exception as e:
                    raise FileNotFoundError(
                        f"Erreur lors de la lecture des secrets: {e}\n"
                        f"Vérifie que tu as ajouté [credentials.usernames.*] dans Streamlit Cloud Secrets"
                    )
        except (ImportError, Exception):
            pass
    
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    # Si le fichier existe en local, le retourner
    if config_path.exists():
        return config_path

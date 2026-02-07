# ==============================================================================
# UTILITAIRES DE GESTION DES VARIABLES D'ENVIRONNEMENT
# ==============================================================================

import os

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

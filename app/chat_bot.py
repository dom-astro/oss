# ==============================================================================
# IMPORTS - Dépendances pour le chatbot astronomique avec RAG (Retrieval-Augmented Generation)
# ==============================================================================

# LangChain - Framework pour construire des applications basées sur les LLM
import re
from langchain_mistralai.embeddings import MistralAIEmbeddings  # Modèle d'embeddings Mistral
from langchain_mistralai.chat_models import ChatMistralAI  # Modèle de chat Mistral
from langchain.chains.combine_documents import create_stuff_documents_chain  # Chaîne de combinaison de documents
from langchain_core.prompts import ChatPromptTemplate  # Template pour les prompts de chat
from langchain.chains import create_retrieval_chain  # Chaîne de récupération augmentée
from langchain_community.vectorstores import FAISS  # Base de données vectorielle FAISS
from langchain_core.prompts import MessagesPlaceholder  # Placeholder pour l'historique du chat
from langchain_core.messages import HumanMessage ,AIMessage  # Messages du chat
from langchain.chains.history_aware_retriever import create_history_aware_retriever  # Retriever conscient de l'historique

# Utilitaires Python
from dotenv import load_dotenv  # Charger les variables d'environnement
from pathlib import Path  # Gestion des chemins fichiers
import os  # Accès aux variables d'environnement
import json  # Manipulation de JSON
import requests  # Requêtes HTTP
from bs4 import BeautifulSoup  # Web scraping
from datetime import datetime, timedelta  # Gestion des dates/heures
from env_utils import get_env_variable  # Gestion centralisée des variables d'environnement

# ==============================================================================
# CACHE GLOBAL - Stockage des données météo/programme du site SkyWatch
# ==============================================================================
# Ce cache évite de récupérer les données trop fréquemment du site SkyWatch
SKYWATCH_CACHE = {
    'data': None,  # Contient les données météo et astronomiques brutes
    'timestamp': None,  # Timestamp de la dernière récupération
    'refresh_interval': 300  # Intervalle de rafraîchissement en secondes (5 minutes)
}

# Cache pour le Top 10 des objets Messier visibles (page publique).
# Même mécanique que SKYWATCH_CACHE : TTL de 5 minutes pour limiter
# les appels réseau au NAS hébergeant catalogue-data.js.
MESSIER_PAGE_CACHE = {
    'data': None,       # Chaîne de texte formatée du Top 10 (ou None si non chargé)
    'rows': [],         # Liste de dicts des 10 objets (utilisée pour l'affichage)
    'timestamp': None,  # datetime de la dernière récupération réussie
    'refresh_interval': 300  # TTL en secondes (5 minutes)
}

def model_and_embedding_function(api_key):
    """
    Initialise le modèle de langage et la fonction d'embedding avec les clés API Mistral.
    
    Args:
        api_key (str): Clé API Mistral pour l'authentification
        
    Returns:
        tuple: (model ChatMistral, fonction d'embeddings)
    """
    # Crée la fonction d'embedding pour transformer le texte en vecteurs
    embedding_function = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)

    # Initialise le modèle de chat LLM Mistral pour générer les réponses
    model = ChatMistralAI(mistral_api_key=api_key, model="mistral-large-latest")
    return model, embedding_function

def should_fetch_skywatch(user_input: str) -> bool:
    """
    Détermine si une question porte sur la météo ou le programme d'observation.
    
    Args:
        user_input (str): Le texte de la question de l'utilisateur
        
    Returns:
        bool: True si la question concerne la météo/conditions du ciel/programme
    """
    # Mots-clés qui indiquent une question sur la météo ou le programme
    keywords = ['météo', 'meteo', 'temps', 'soir', 'programme', 'ce soir', 'pluie', 'nuages', 'ciel', 'conditions', 'beau', 'observation', 'sky', 'weather', 'sky watch', 'skywatch', 'nuit', 'seeing', 'transparence', 'couverture', 'couverture nuageuse', 'observer', 'visible', 'visibilité']
    user_lower = user_input.lower()
    
    # Vérifie si au moins un mot-clé est présent dans la question
    should_fetch = any(keyword in user_lower for keyword in keywords)
    
    if should_fetch:
        print(f"INFO - Récupération SkyWatch déclenchée pour: {user_input}")
    
    return should_fetch

def should_use_messier_catalog(user_input: str) -> bool:
    """
    Détermine si la question porte sur le catalogue Messier d'objets astronomiques.
    
    Args:
        user_input (str): Le texte de la question de l'utilisateur
        
    Returns:
        bool: True si la question concerne les objets Messier
    """
    # Mots-clés qui indiquent une question sur les objets Messier
    keywords = ['messier', 'catalogue messier', 'objets messier', 'objets de messier', 'objet messier', 'objet de messier', ' m31', ' m42', ' m45', ' m13', ' m1 ', 'objets m ']
    user_lower = user_input.lower()
    
    # Vérifie si au moins un mot-clé Messier est présent
    should_use = any(keyword in user_lower for keyword in keywords)
    
    if should_use:
        print(f"INFO - Utilisation du catalogue Messier déclenchée pour: {user_input}")
    
    return should_use

def should_fetch_messier_page(user_input: str) -> bool:
    """
    Détermine si la question porte spécifiquement sur les objets Messier
    visibles ce soir, ce qui nécessite de consulter la page publique du catalogue
    plutôt que le PDF statique.

    Combine deux conditions :
    1. La question concerne le catalogue Messier (should_use_messier_catalog).
    2. Elle mentionne la visibilité ou la soirée ("visible", "visibles", "ce soir").

    Args:
        user_input (str): Le texte de la question de l'utilisateur.

    Returns:
        bool: True si les deux conditions sont réunies, False sinon.
    """
    user_lower = user_input.lower()
    return should_use_messier_catalog(user_input) and (
        "visible" in user_lower or "visibles" in user_lower or "ce soir" in user_lower
    )

def _get_french_label(value: str) -> str:
    """
    Extrait le libellé français d'une chaîne bilangue du format "English/Français".

    Dans le fichier catalogue-data.js, certains champs (comme "objet" ou "saison")
    sont encodés sous la forme "English label/Libellé français". Cette fonction
    retourne la partie droite (française) si le séparateur "/" est présent,
    ou la valeur brute sinon.

    Exemples :
        "Open Cluster/Amas ouvert"  →  "Amas ouvert"
        "Winter/Hiver"              →  "Hiver"
        "Nébuleuse"                 →  "Nébuleuse"  (pas de "/")

    Args:
        value (str): Valeur brute issue du catalogue JS.

    Returns:
        str: Libellé français (ou valeur entière si pas de "/"), chaîne vide si None.
    """
    if not value:
        return ""
    parts = str(value).split("/")
    if len(parts) > 1:
        return parts[1].strip()
    return str(value).strip()

def _season_for_date(date_value: datetime) -> str:
    """
    Retourne la saison astronomique française correspondant à une date donnée.

    Le découpage est basé sur les mois calendaires (approximation courante) :
        Décembre, Janvier, Février  →  Hiver
        Mars, Avril, Mai            →  Printemps
        Juin, Juillet, Août         →  Été
        Septembre, Octobre, Novembre→  Automne

    Note : on soustrait 1 au mois pour utiliser un index de 0 à 11,
    ce qui simplifie la condition du mois de décembre (index 11 → 11 % 12 = 11).

    Args:
        date_value (datetime): La date à analyser.

    Returns:
        str: L'une des valeurs "Hiver", "Printemps", "Été" ou "Automne".
    """
    month = date_value.month - 1  # mois 0-indexé (0=Jan … 11=Déc)
    if month in (11, 0, 1):
        return "Hiver"
    if 2 <= month <= 4:
        return "Printemps"
    if 5 <= month <= 7:
        return "Été"
    return "Automne"

def _is_visible_by_site(mag_value, saison_value: str) -> bool:
    """
    Détermine si un objet Messier est considéré comme observable depuis le site
    de l'observatoire, selon deux critères indépendants :

    1. **Magnitude** : l'objet doit avoir une magnitude ≤ 6 (limite de l'œil nu
       sous un ciel correct, seuil choisi pour garantir une observabilité minimale).
    2. **Saison** : si un champ saison est renseigné dans le catalogue, il doit
       correspondre à la saison astronomique actuelle (calculée par _season_for_date).
       Si le champ est vide, le critère saison est ignoré.

    Args:
        mag_value: Magnitude apparente de l'objet (str ou float convertible).
        saison_value (str): Valeur de saison issue du catalogue JS, potentiellement
                            au format "English/Français" (ex. "Winter/Hiver").

    Returns:
        bool: True si l'objet est à la fois suffisamment brillant ET de saison.
    """
    if mag_value is None:
        return False
    try:
        mag_ok = float(mag_value) <= 6
    except (TypeError, ValueError):
        return False
    if not saison_value:
        return mag_ok  # Pas de contrainte saisonnière : critère magnitude seul

    label = _get_french_label(saison_value).lower()
    today_season = _season_for_date(datetime.now())

    if "hiver" in label and today_season != "Hiver":
        return False
    if "print" in label and today_season != "Printemps":
        return False
    if ("été" in label or "ete" in label) and today_season != "Été":
        return False
    if "automne" in label and today_season != "Automne":
        return False
    return mag_ok

def _get_constellation_display(obj: dict) -> str:
    """
    Retourne le nom de constellation le plus lisible disponible pour un objet Messier.

    Le catalogue JS peut stocker le nom de constellation sous plusieurs clés selon
    la version des données. On essaie les clés dans l'ordre de préférence :
        1. "nom_francais"          → nom français explicite (ex. "Orion")
        2. "latin_name_nom_latin"  → nom latin (ex. "Orion")
        3. "const"                 → abréviation IAU (ex. "Ori")

    Args:
        obj (dict): Dictionnaire représentant un objet du catalogue Messier.

    Returns:
        str: Nom de constellation (chaîne vide si aucune clé n'est trouvée).
    """
    return obj.get("nom_francais") or obj.get("latin_name_nom_latin") or obj.get("const") or ""

def _fetch_messier_catalog_data() -> list:
    """
    Télécharge et parse le fichier JavaScript catalogue-data.js hébergé sur le NAS.

    Le fichier contient une déclaration JS de la forme :
        const messierData = [ {...}, {...}, ... ];

    On extrait le tableau JSON embarqué à l'aide d'une expression régulière,
    puis on le désérialise avec json.loads.

    Returns:
        list: Liste de dicts représentant les 110 objets Messier,
              ou liste vide en cas d'erreur (réseau, parsing, etc.).
    """
    url = "http://nas-gdl2.synology.me/messier/catalogue-data.js"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        text = response.text
        # Extrait le tableau JS entre crochets (flags=re.S pour matcher les sauts de ligne)
        match = re.search(r"const\s+messierData\s*=\s*(\[.*\])\s*;?", text, flags=re.S)
        if not match:
            return []
        return json.loads(match.group(1))
    except Exception as e:
        print(f"WARNING - Impossible de charger catalogue-data.js: {e}")
        return []

def get_messier_context(vector, doc_id: str, max_chunks: int = 110):
    """
    Récupère tous les chunks FAISS appartenant au document "Catalogue Messier"
    identifié par son doc_id.

    Stratégie en deux passes :
    1. **Scan direct du docstore** : parcourt tous les vecteurs et filtre ceux dont
       la métadonnée "doc_id" correspond. C'est la méthode la plus fiable car elle
       ne dépend pas de la similarité sémantique.
    2. **Fallback par similarité** : si le scan direct ne retourne rien (index vide
       ou metadata absente), effectue une recherche sémantique approximative sur
       "Catalogue Messier objets M" et filtre les résultats par doc_id.

    Args:
        vector: Base de données vectorielle FAISS (peut être None).
        doc_id (str): UUID du document Messier tel qu'enregistré dans document_index.json.
        max_chunks (int): Nombre maximum de chunks à retourner (défaut 110,
                          soit un chunk par objet Messier au maximum).

    Returns:
        list: Liste de Documents LangChain (page_content + metadata),
              liste vide si vector est None, doc_id vide, ou en cas d'erreur.
    """
    if vector is None or not doc_id:
        return []

    try:
        # Passe 1 : scan direct — on parcourt le mapping index → docstore_id
        docs = []
        for ds_id in vector.index_to_docstore_id.values():
            doc = vector.docstore.search(ds_id)
            if getattr(doc, "metadata", {}).get("doc_id") == doc_id:
                docs.append(doc)
                if len(docs) >= max_chunks:
                    break
        
        if docs:
            print(f"INFO - Retrieved {len(docs)} chunks from Catalogue Messier")
            return docs
        
        # Passe 2 : fallback par similarité sémantique
        try:
            candidates = vector.similarity_search("Catalogue Messier objets M", k=50)
            messier_docs = [doc for doc in candidates if getattr(doc, "metadata", {}).get("doc_id") == doc_id]
            return messier_docs[:max_chunks]
        except Exception as e:
            print(f"WARNING - Similarity search failed: {e}")
            return []
    except Exception as e:
        print(f"WARNING - Failed to load Messier context: {e}")
        return []

def load_messier_images_from_assets(max_images: int = 5) -> list:
    """
    Load ALL Messier object images from assets/images/catalogue_messier folder.
    Images should be named M-001.jpg, M-002.jpg, ... M-110.jpg
    Note: max_images is ignored - all images are loaded to support filtering by mention in answer
    
    Args:
        max_images (int): Ignored - loads all available images
    
    Returns:
        list: List of dicts with 'image_path', 'messier_number', 'source'
    """
    images_data = []
    
    try:
        assets_path = Path(__file__).resolve().parent.parent / "assets" / "images" / "catalogue_messier"
        
        if not assets_path.exists():
            print(f"WARNING - Messier images folder not found at {assets_path}")
            return images_data
        
        # Get all image files (jpg, png, jpeg)
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']:
            image_files.extend(sorted(assets_path.glob(ext)))
        
        # Sort by filename to get M-001, M-002, etc. in order (load ALL, not just 5)
        image_files = sorted(image_files)
        
        for img_path in image_files:
            try:
                # Extract Messier number from filename (e.g., M-001 -> M1)
                filename = img_path.stem  # Get filename without extension
                messier_number = None
                if filename.upper().startswith("M-"):
                    try:
                        # Extract number and handle leading zeros (M-001 -> 1, M-042 -> 42)
                        messier_number = int(filename.split("-")[1].lstrip("0") or "0")
                    except Exception:
                        messier_number = None
                
                images_data.append({
                    'image_path': str(img_path),
                    'messier_label': filename,  # M-001, M-002, etc.
                    'messier_number': messier_number,
                    'source': 'Catalogue Messier'
                })
                print(f"INFO - Loaded image: {filename}")
                
            except Exception as e:
                print(f"WARNING - Could not process image {img_path}: {e}")
                continue
        
        print(f"INFO - Total Messier images loaded: {len(images_data)}")
        
    except Exception as e:
        print(f"ERROR - Could not load Messier images: {e}")
    
    return images_data

def fetch_messier_page_top10() -> tuple:
    """
    Retourne les 10 objets Messier les plus faciles à observer ce soir,
    triés par magnitude croissante (les plus brillants en premier).

    Processus :
    1. Vérifie si le cache MESSIER_PAGE_CACHE est encore valide (< 5 min).
       Si oui, retourne directement les données mises en cache.
    2. Charge le catalogue complet via _fetch_messier_catalog_data().
    3. Filtre les objets selon _is_visible_by_site() : magnitude ≤ 6 ET saison courante.
    4. Trie par magnitude croissante et retient les 10 premiers.
    5. Construit une chaîne de texte formatée (pour injection dans le prompt LLM)
       et met à jour le cache.

    Returns:
        tuple:
            - str  : Texte formaté "LISTE DES 10 OBJETS MESSIER AFFICHÉS…"
                     ou message d'erreur si le catalogue est indisponible.
            - list : Liste de dicts (messier, ngc, objet, saison, mag,
                     constellation, ra, dec, dimension, visible),
                     ou liste vide en cas d'échec.
    """
    global MESSIER_PAGE_CACHE

    now = datetime.now()
    # Retour immédiat si les données en cache sont encore fraîches
    if MESSIER_PAGE_CACHE['data'] is not None and MESSIER_PAGE_CACHE['timestamp'] is not None:
        if (now - MESSIER_PAGE_CACHE['timestamp']).total_seconds() < MESSIER_PAGE_CACHE['refresh_interval']:
            return MESSIER_PAGE_CACHE['data'], MESSIER_PAGE_CACHE.get('rows', [])

    url = "http://messier.astronomie-pointedudiable.fr/"
    try:
        data = _fetch_messier_catalog_data()
        if not data:
            return "Impossible de récupérer les données Messier (catalogue-data.js indisponible).", []

        filtered = [obj for obj in data if _is_visible_by_site(obj.get("mag"), obj.get("saison"))]
        if not filtered:
            return "Impossible de récupérer les objets Messier (aucun visible selon les filtres).", []

        def mag_sort_value(obj):
            """Clé de tri : retourne la magnitude en float, ou 0.0 si la conversion échoue."""
            try:
                return float(obj.get("mag", 0))
            except (TypeError, ValueError):
                return 0.0

        sorted_objects = sorted(filtered, key=mag_sort_value)
        objects = []
        for obj in sorted_objects[:10]:
            objects.append({
                "messier": obj.get("messier", ""),
                "ngc": obj.get("ngc"),
                "objet": _get_french_label(obj.get("objet", "")),
                "saison": _get_french_label(obj.get("saison", "")),
                "mag": obj.get("mag", "?"),
                "constellation": _get_constellation_display(obj),
                "ra": obj.get("ra"),
                "dec": obj.get("dec"),
                "dimension": obj.get("dimension"),
                "visible": "O",
            })

        lines = [
            "LISTE DES 10 OBJETS MESSIER AFFICHÉS (source: http://messier.astronomie-pointedudiable.fr/)",
        ]
        for idx, obj in enumerate(objects, 1):
            lines.append(
                f"{idx}. {obj['messier']} | {obj['objet']} | Saison: {obj['saison']} | Mag: {obj['mag']} | Constellation: {obj['constellation']} | Visible: {obj['visible']}"
            )

        content = "\n".join(lines)
        MESSIER_PAGE_CACHE['data'] = content
        MESSIER_PAGE_CACHE['rows'] = objects
        MESSIER_PAGE_CACHE['timestamp'] = now
        return content, objects
    except Exception as e:
        return f"Impossible de récupérer la page Messier: {e}", []

def parse_messier_page_top10(messier_page_content: str) -> list:
    """
    Parse la chaîne de texte générée par fetch_messier_page_top10() et en extrait
    une liste de dicts structurés, utilisables pour l'affichage en tableau.

    La chaîne attendue contient des lignes numérotées au format :
        "N. MXX | Type | Saison: X | Mag: X.X | Constellation: X | Visible: O"

    Args:
        messier_page_content (str): Texte brut retourné par fetch_messier_page_top10().

    Returns:
        list: Liste de dicts avec les clés "messier", "objet", "saison",
              "mag", "constellation", "visible". Liste vide si le contenu
              est absent ou ne contient pas l'en-tête attendu.
    """
    if not messier_page_content or "LISTE DES 10 OBJETS" not in messier_page_content:
        return []

    rows = []
    for line in messier_page_content.splitlines():
        # Détecte les lignes numérotées de 1. à 10.
        if line.strip().startswith(tuple(str(i) + "." for i in range(1, 11))):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 6:
                # Sépare le numéro de ligne ("1.") du label Messier ("M 42")
                idx_and_m = parts[0].split(".", 1)
                messier_label = idx_and_m[1].strip() if len(idx_and_m) > 1 else parts[0].strip()
                rows.append({
                    "messier": messier_label,
                    "objet": parts[1],
                    "saison": parts[2].replace("Saison:", "").strip(),
                    "mag": parts[3].replace("Mag:", "").strip(),
                    "constellation": parts[4].replace("Constellation:", "").strip(),
                    "visible": parts[5].replace("Visible:", "").strip(),
                })
    return rows

def _parse_magnitude(value: str):
    """
    Convertit une chaîne de magnitude en float, en gérant la virgule décimale.

    Args:
        value (str): Magnitude sous forme de chaîne (ex. "6,4" ou "6.4").

    Returns:
        float | None: Valeur numérique, ou None si la conversion échoue.
    """
    if not value:
        return None
    try:
        return float(value.replace(",", ".").strip())
    except ValueError:
        return None

def _is_visible(value: str) -> bool:
    """
    Indique si une valeur de visibilité booléenne est considérée comme vraie.

    Accepte plusieurs représentations courantes : "oui", "yes", "true", "1", "o".
    La comparaison est insensible à la casse et aux espaces.

    Args:
        value (str): Valeur brute du champ visibilité.

    Returns:
        bool: True si la valeur correspond à "visible/oui", False sinon.
    """
    if not value:
        return False
    normalized = value.strip().lower()
    return normalized in {"oui", "yes", "true", "1", "o"}

def _extract_messier_number(label: str):
    """
    Extrait le numéro entier d'un label Messier sous divers formats.

    Formats reconnus : "M31", "M 31", "M-31", "M-031", "M 031", etc.
    Le pattern regex est insensible à la casse et tolère des espaces ou tirets
    entre "M" et les chiffres.

    Args:
        label (str): Label brut (ex. "M-042", "M 1", "m31").

    Returns:
        int | None: Numéro Messier (1–110), ou None si non trouvé ou invalide.
    """
    if not label:
        return None
    match = re.search(r"\bM\s*-?\s*(\d{1,3})\b", label, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None

def _visibility_label(mag_value) -> str:
    """Convertit une magnitude en étiquette : 'Facile' (≤4), 'Modérée' (≤6) ou 'Difficile'."""
    try:
        mag = float(mag_value)
    except (TypeError, ValueError):
        return "Difficile"
    if mag <= 4:
        return "Facile"
    if mag <= 6:
        return "Modérée"
    return "Difficile"

def _photographiable_label(obj_type: str) -> str:
    """Retourne 'Non' pour les étoiles doubles (peu adaptées à l'astrophoto), 'Oui' sinon."""
    if not obj_type:
        return "Oui"
    lowered = obj_type.lower()
    if "étoile double" in lowered:
        return "Non"
    return "Oui"

def _describe_object(obj_type: str) -> str:
    """
    Retourne une courte description visuelle adaptée au type d'objet Messier.

    Ces descriptions sont affichées dans la réponse du chatbot pour donner à
    l'observateur un aperçu de ce qu'il verra à l'oculaire.

    Args:
        obj_type (str): Type de l'objet en français (ex. "Galaxie", "Amas ouvert").

    Returns:
        str: Phrase descriptive adaptée au type, ou description générique si inconnu.
    """
    if not obj_type:
        return "Objet du catalogue Messier, intéressant pour l'observation visuelle."
    lowered = obj_type.lower()
    if "nébuleuse" in lowered:
        return "Nuage de gaz et de poussières, souvent riche en détails visibles à faible grossissement."
    if "galaxie" in lowered:
        return "Galaxie lointaine dont la structure devient plus visible sous un ciel sombre."
    if "amas ouvert" in lowered:
        return "Amas d'étoiles jeunes et dispersées, agréable à observer au grand champ."
    if "amas globulaire" in lowered:
        return "Amas sphérique très dense d'étoiles anciennes, spectaculaire à grossissement moyen."
    if "reste de supernova" in lowered:
        return "Vestige d'une explosion stellaire, souvent riche en filaments ténus."
    if "étoile double" in lowered:
        return "Système de deux étoiles visibles à l'oculaire avec une bonne résolution."
    return "Objet du catalogue Messier, intéressant pour l'observation visuelle."
def _describe_object(obj_type: str) -> str:
    """
    Retourne une courte description visuelle adaptée au type d'objet Messier.

    Ces descriptions sont affichées dans la réponse du chatbot pour donner à
    l'observateur un aperçu de ce qu'il verra à l'oculaire.

    Args:
        obj_type (str): Type de l'objet en français (ex. "Galaxie", "Amas ouvert").

    Returns:
        str: Phrase descriptive adaptée au type, ou description générique si inconnu.
    """
    if not obj_type:
        return "Objet du catalogue Messier, intéressant pour l’observation visuelle."
    lowered = obj_type.lower()
    if "nébuleuse" in lowered:
        return "Nuage de gaz et de poussières, souvent riche en détails visibles à faible grossissement."
    if "galaxie" in lowered:
        return "Galaxie lointaine dont la structure devient plus visible sous un ciel sombre."
    if "amas ouvert" in lowered:
        return "Amas d’étoiles jeunes et dispersées, agréable à observer au grand champ."
    if "amas globulaire" in lowered:
        return "Amas sphérique très dense d’étoiles anciennes, spectaculaire à grossissement moyen."
    if "reste de supernova" in lowered:
        return "Vestige d’une explosion stellaire, souvent riche en filaments ténus."
    if "étoile double" in lowered:
        return "Système de deux étoiles visibles à l’oculaire avec une bonne résolution."
    return "Objet du catalogue Messier, intéressant pour l’observation visuelle."

def _has_interest(obj_type: str, mag_value, dimension_value: str) -> bool:
    """
    Détermine si un objet Messier présente un intérêt particulier à mettre en valeur.

    Un objet est jugé "intéressant" si au moins l'une de ces conditions est vraie :
    - Sa magnitude est ≤ 3 (très brillant, facilement visible à l'œil nu).
    - Son type est une nébuleuse ou une galaxie (objets visuellement riches).
    - Son champ "dimension" contient des chiffres (taille angulaire renseignée).

    Args:
        obj_type (str): Type de l'objet (ex. "Nébuleuse", "Galaxie").
        mag_value: Magnitude (convertible en float).
        dimension_value (str): Taille angulaire brute (ex. "90'x40'").

    Returns:
        bool: True si au moins un critère d'intérêt est satisfait.
    """
    try:
        mag = float(mag_value)
    except (TypeError, ValueError):
        mag = None
    if mag is not None and mag <= 3:
        return True
    if obj_type and ("nébuleuse" in obj_type.lower() or "galaxie" in obj_type.lower()):
        return True
    if dimension_value and any(ch.isdigit() for ch in str(dimension_value)):
        return True
    return False

def _interest_text(obj_type: str) -> str:
    """
    Retourne un texte de mise en valeur adapté au type d'objet Messier.

    Contrairement à _describe_object() qui décrit ce que l'on voit à l'oculaire,
    cette fonction produit un conseil d'observation (filtre recommandé, intérêt
    spécifique) pour encourager l'observateur à pointer cet objet.

    Args:
        obj_type (str): Type de l'objet en français (ex. "Nébuleuse", "Amas globulaire").

    Returns:
        str: Conseil ou argument d'observation, ou texte générique si type inconnu.
    """
    if not obj_type:
        return "Objet emblématique et facile à repérer au grand champ."
    lowered = obj_type.lower()
    if "nébuleuse" in lowered:
        return "Contrastes intéressants et structures fines, idéal avec un filtre adapté."
    if "galaxie" in lowered:
        return "Intéressante pour comparer le halo et la structure sous un ciel sombre."
    if "amas ouvert" in lowered:
        return "Bel aspect en grand champ, parfait pour la photographie."
    if "amas globulaire" in lowered:
        return "Cœur dense spectaculaire, bon test de résolution."
    if "reste de supernova" in lowered:
        return "Structures filamenteuses remarquables, meilleur rendu avec filtre OIII."
    return "Objet emblématique et facile à repérer au grand champ."

def format_messier_page_response(rows: list) -> str:
    """
    Formate une liste d'objets Messier en blocs de texte markdown pour la réponse du chatbot.

    Pour chaque objet, un bloc multi-lignes est généré avec :
    - Identifiant et type (ex. "M42 - Nébuleuse")
    - Caractéristiques : constellation, magnitude, taille angulaire
    - Niveau de visibilité : Facile / Modérée / Difficile (basé sur la magnitude)
    - Photographiable : Oui / Non
    - Conseil d'observation personnalisé
    - Description visuelle synthétique

    Args:
        rows (list): Liste de dicts produite par fetch_messier_page_top10().
                     Chaque dict contient les clés : messier, objet, constellation,
                     mag, dimension, ngc, ra, dec.

    Returns:
        str: Blocs markdown séparés par des lignes vides ("\n\n"),
             ou message d'erreur si rows est vide.
    """
    if not rows:
        return "Je n'ai pas pu extraire les 10 objets Messier depuis la page publique."

    blocks = []
    for obj in rows:
        messier_label = obj.get("messier", "M-XX")
        ngc_value = obj.get("ngc") or "—"
        obj_type = obj.get("objet", "—")
        ra = obj.get("ra") or "—"
        dec = obj.get("dec") or "—"
        constellation = obj.get("constellation") or "—"
        dimension = obj.get("dimension") or "—"
        magnitude = obj.get("mag") if obj.get("mag") is not None else "—"
        visibility = _visibility_label(obj.get("mag"))
        photo = _photographiable_label(obj_type)
        description = _describe_object(obj_type)

        visibility_line = "Visibilité: Ce soir (voir SkyWatch pour les horaires)."
        if visibility in {"Facile", "Modérée", "Difficile"}:
            visibility_line = f"Visibilité: {visibility} (selon la magnitude)."

        conseil = "Observable à l’œil nu ou aux jumelles." if visibility == "Facile" else "Observer avec un instrument adapté."
        if photo == "Oui":
            conseil = f"{conseil} Idéal pour la photo grand champ." if "grand champ" not in conseil else conseil

        block_lines = [
            f"{messier_label} - {obj_type}",
            f"Type: {obj_type} | Constellation : {constellation} | Magnitude : {magnitude} | Taille : {dimension}",
            visibility_line,
            f"Photographiable : {photo}",
            f"Conseil: {conseil}",
            f"Description: {description}",
        ]

        blocks.append("\n".join(block_lines))

    return "\n\n".join(blocks)

def build_messier_images_for_rows(rows: list) -> list:
    """
    Retourne les entrées d'images correspondant aux objets d'une liste de rows Messier.

    Charge toutes les images disponibles depuis assets/ (via load_messier_images_from_assets),
    extrait les numéros Messier des rows, puis filtre et trie les images dans le même
    ordre que les rows — ce qui garantit l'alignement image/texte dans l'interface.

    Args:
        rows (list): Liste de dicts produite par fetch_messier_page_top10().
                     Chaque dict doit contenir la clé "messier" (ex. "M 42").

    Returns:
        list: Liste de dicts image dans l'ordre des rows, avec les clés :
              'image_path', 'messier_label', 'messier_number', 'source'.
              Liste vide si aucune image n'est disponible ou si rows est vide.
    """
    images = load_messier_images_from_assets()
    if not images:
        return []

    wanted_numbers = []
    for row in rows:
        num = _extract_messier_number(row.get("messier"))
        if num and num not in wanted_numbers:
            wanted_numbers.append(num)

    if not wanted_numbers:
        return []

    matched = [img for img in images if img.get("messier_number") in wanted_numbers]
    matched.sort(key=lambda img: wanted_numbers.index(img.get("messier_number")))
    return matched

def create_messier_page_document(messier_page_content: str):
    """
    Encapsule le contenu textuel du Top 10 Messier dans un Document LangChain.

    Ce document est injecté en tête de la liste des sources retournées à interface.py,
    ce qui permet à l'expander "Documents utilisés" d'afficher l'origine des données
    Messier et à la chaîne RAG d'y accéder si nécessaire.

    Args:
        messier_page_content (str): Texte formaté retourné par fetch_messier_page_top10().

    Returns:
        Document: Document LangChain avec métadonnées source et type.
    """
    from langchain_core.documents import Document
    return Document(
        page_content=messier_page_content,
        metadata={
            'source': 'messier.astronomie-pointedudiable.fr',
            'type': 'messier_page_top10'
        }
    )

def find_messier_info(messier_number: int, messier_docs: list) -> str:
    """
    Recherche et retourne le chunk FAISS le plus pertinent pour un numéro Messier donné.

    La fonction essaie de nombreux formats de notation (M31, M 31, M-031, Messier 31…)
    et choisit le chunk où la correspondance apparaît le plus tôt dans le texte —
    ce qui favorise les chunks contenant l'en-tête de l'objet plutôt qu'une simple mention.

    Utilisée pour enrichir l'affichage des images Messier avec les informations
    issues du PDF "Catalogue Messier.pdf".

    Args:
        messier_number (int): Numéro Messier (1–110).
        messier_docs (list): Liste de Documents LangChain issus du catalogue.

    Returns:
        str: Extrait du meilleur chunk trouvé (max 1200 caractères),
             ou chaîne vide si aucune correspondance n'est trouvée.
    """
    if not messier_number or not messier_docs:
        return ""

    # Build a comprehensive list of patterns to search for
    patterns = [
        f"M {messier_number} ",          # M 31 with space after
        f"M{messier_number} ",           # M31 with space after
        f"M -{messier_number}",          # M -31
        f"M-{messier_number:03d}",       # M-031
        f"M {messier_number:03d}",       # M 031
        f"M{messier_number:03d}",        # M031
        f"{messier_number}.",            # Just number with period (e.g., "31.")
        f"M {messier_number:02d}",       # M 31 (2-digit)
        f"M{messier_number:02d}",        # M31 (2-digit)
        f"M {messier_number}\n",         # M 31 at line end
    ]
    
    best_match = ""
    best_position = float('inf')  # Track where in the text the match was found
    
    for doc in messier_docs:
        text = doc.page_content if hasattr(doc, "page_content") else str(doc)
        text_lower = text.lower()
        
        # Check if any pattern matches
        for pat in patterns:
            pat_lower = pat.lower()
            pos = text_lower.find(pat_lower)
            if pos >= 0:
                # Prefer matches found earlier in the chunk (likely the header)
                if pos < best_position:
                    best_match = text
                    best_position = pos
                    break  # Found a match in this doc, move to next doc
    
    # Return up to 1200 chars for better information display
    return best_match[:1200] if best_match else ""

def extract_messier_numbers(text: str) -> list:
    """
    Extrait tous les numéros Messier présents dans un texte, dans leur ordre d'apparition.

    Gère de nombreux formats courants : M1, M 1, M-1, M01, M001, (M1), Messier 31, etc.
    Les doublons sont éliminés (première occurrence conservée) et seuls les numéros
    valides entre 1 et 110 sont retenus.

    Cette fonction est utilisée pour identifier quelles images Messier afficher
    en fonction des objets mentionnés dans la réponse du LLM.

    Args:
        text (str): Texte quelconque (réponse du chatbot, question, etc.).

    Returns:
        list[int]: Liste ordonnée de numéros Messier uniques (1–110),
                   liste vide si text est vide ou contient aucune mention.
    """
    if not text:
        return []
    
    # Multiple patterns to catch different formats: M1, M 1, M-1, M01, M001, (M1), M 31, etc.
    patterns = [
        r"\bM\s*-?\s*(\d{1,3})\b",      # M 31, M31, M-31, M 031, etc.
        r"\(M\s*(\d{1,3})\)",           # (M 31)
        r"Messier\s+(\d{1,3})",         # Messier 31
        r"M\d+",                         # Catch M followed by digits anywhere
    ]
    
    numbers = []
    for pattern_str in patterns:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        for match in pattern.finditer(text):
            try:
                # Extract the number - handle both direct match and group(1)
                if match.groups():
                    num = int(match.group(1))
                else:
                    # Extract digits from the match
                    match_str = match.group(0)
                    digits = ''.join(c for c in match_str if c.isdigit())
                    num = int(digits)
                
                if 1 <= num <= 110 and num not in numbers:
                    numbers.append(num)
            except (ValueError, IndexError, AttributeError):
                continue
    
    return numbers

def fetch_skywatch_data() -> str:
    """
    Récupère les données météo et astronomiques du site SkyWatch avec système de cache.
    
    Le cache a une durée de vie de 5 minutes pour éviter les requêtes inutiles au site.
    En cas d'erreur de connexion, utilise le cache ancien s'il existe.
    
    Returns:
        str: Les données météo et astronomiques formatées
    """
    global SKYWATCH_CACHE
    
    # Vérifier si les données en cache sont encore valides (moins de 5 minutes)
    now = datetime.now()
    if SKYWATCH_CACHE['data'] is not None and SKYWATCH_CACHE['timestamp'] is not None:
        age = (now - SKYWATCH_CACHE['timestamp']).total_seconds()
        if age < SKYWATCH_CACHE['refresh_interval']:
            print(f"INFO - Utilisation des données SkyWatch en cache (âge: {int(age)}s)")
            return SKYWATCH_CACHE['data']
        else:
            print(f"INFO - Cache expiré (âge: {int(age)}s), rafraîchissement...")
    
    try:
        # URLs du site SkyWatch (avec redirection possible)
        urls = [
            "http://nas-gdl2.synology.me/skywatch/",
            "http://skywatch.astronomie-pointedudiable.fr/"
        ]
        
        # Headers HTTP pour se présenter comme un navigateur
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Essayer de se connecter à chaque URL
        response = None
        for url in urls:
            try:
                response = requests.get(url, timeout=10, headers=headers, allow_redirects=True)
                print(f"DEBUG - Connexion à {url}, statut: {response.status_code}")
                if response.status_code == 200:
                    break
            except Exception as e:
                print(f"DEBUG - Échec pour {url}: {e}")
                continue
        
        if not response:
            # Si la connexion échoue, utiliser le cache ancien s'il existe
            if SKYWATCH_CACHE['data']:
                print("ATTENTION - Connexion échouée, utilisation du cache ancien")
                return SKYWATCH_CACHE['data']
            return "Impossible de se connecter au site SkyWatch"
        
        # Parse le contenu HTML du site avec BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extrait tout le contenu texte (le site n'utilise pas de structure HTML claire)
        all_text = soup.get_text(separator='\n', strip=True)
        lines = [l.strip() for l in all_text.split('\n') if l.strip()]
        
        # Prépare la liste des données extraites
        extracted_data = []
        extracted_data.append("=== DONNÉES MÉTÉO ET ASTRONOMIQUES EN TEMPS RÉEL DU SITE SKYWATCH ===\n")
        extracted_data.append(f"Dernière mise à jour : {now.strftime('%d/%m/%Y %H:%M:%S')}\n")
        
        # Mots-clés des données météo à extraire
        weather_keywords = {
            'Date', 'Heure', 'Température', 'Vent', 'Humidité', 
            'Lever du soleil', 'Coucher du soleil', 'Qualité du ciel',
            'Météo', 'Description', 'Ville'
        }
        
        extracted_data.append("\n** MÉTÉO ACTUELLE **")
        # Parcourt les lignes pour extraire les données météo
        i = 0
        while i < len(lines):
            line = lines[i]
            # Vérifie si cette ligne contient un mot-clé météo
            if any(keyword in line for keyword in weather_keywords):
                # La valeur est généralement à la ligne suivante
                if i + 1 < len(lines):
                    value = lines[i + 1]
                    # Vérifier que la valeur n'est pas un autre mot-clé
                    if not any(kw in value for kw in weather_keywords) and len(value) < 50:
                        extracted_data.append(f"{line}: {value}")
                        i += 2  # Ignore les deux lignes traitées
                        continue
            i += 1
        
        # Extrait les données des éphémérides planétaires
        extracted_data.append("\n** ÉPHÉMÉRIDES DES PLANÈTES **")
        planet_names = ['Mercure', 'Vénus', 'Mars', 'Jupiter', 'Saturne', 'Uranus', 'Neptune']
        
        # Pour chaque planète, extrait ses données (heures de lever, etc.)
        for planet in planet_names:
            if planet in lines:
                idx = lines.index(planet)
                extracted_data.append(f"{planet}:")
                # Récupère les données après le nom de la planète
                for j in range(idx + 1, min(idx + 5, len(lines))):
                    if lines[j] and len(lines[j]) < 30 and not any(p in lines[j] for p in planet_names):
                        extracted_data.append(f"  {lines[j]}")
        
        # Si les données ont été bien extraites, les mettre en cache
        if len(extracted_data) > 5:
            result = '\n'.join(extracted_data[:50])
            # Stocke les données dans le cache global
            SKYWATCH_CACHE['data'] = result
            SKYWATCH_CACHE['timestamp'] = now
            print(f"INFO - Données SkyWatch mises en cache ({len(extracted_data)} champs)")
            return result
        
        # Plan de secours: extraire le texte pertinent directement
        print("DEBUG - Utilisation de l'extraction de secours")
        # Trouve la section avec les données météo (généralement après le titre "Skywatch")
        start_idx = 0
        for i, line in enumerate(lines):
            if 'Date' in line or 'Heure' in line:
                start_idx = max(0, i - 5)
                break
        
        weather_section = lines[start_idx:start_idx + 60]
        result = '\n'.join(['=== DONNÉES SKYWATCH ==='] + weather_section)
        
        # Mettre à jour le cache même avec le plan de secours
        SKYWATCH_CACHE['data'] = result
        SKYWATCH_CACHE['timestamp'] = now
        
        return result
            
    except Exception as e:
        print(f"ERREUR - Erreur lors du web scraping: {str(e)}")
        # En cas d'erreur, utiliser les données en cache si disponibles
        if SKYWATCH_CACHE['data']:
            print("ATTENTION - Erreur détectée, utilisation du cache")
            return SKYWATCH_CACHE['data']
        import traceback
        traceback.print_exc()
        return f"Impossible de récupérer les données du site: {str(e)}"

def create_skywatch_document(skywatch_content: str):
    """
    Crée un Document LangChain à partir des données SkyWatch.
    
    Cela permet d'intégrer les données météo en temps réel dans la chaîne RAG.
    
    Args:
        skywatch_content (str): Contenu brut des données SkyWatch
        
    Returns:
        Document: Document LangChain avec métadonnées
    """
    from langchain_core.documents import Document
    return Document(
        page_content=skywatch_content,
        metadata={
            'source': 'skywatch.astronomie-pointedudiable.fr',
            'type': 'realtime_weather'  # Identifiant de type pour le filtrage dans interface.py
        }
    )

def create_contextualize_q_system_prompt():
    """
    Crée un prompt pour reformuler les questions en tenant compte de l'historique du chat.
    
    Ce prompt aide le modèle à comprendre les questions qui font référence aux messages précédents,
    en les reformulant comme des questions autonomes.
    
    Returns:
        ChatPromptTemplate: Template de prompt pour la contextualisation des questions
    """
    # Prompt système pour reformuler les questions en tenant compte de l'historique
    contextualize_q_system_prompt = (
        "Étant donné un historique de chat et la dernière question de l'utilisateur "
        "qui pourrait faire référence au contexte de l'historique, "
        "formule une question autonome qui peut être comprise "
        "sans l'historique du chat. Ne réponds PAS à la question, "
        "reformule-la simplement si nécessaire, sinon retourne-la telle quelle."
    )

    # Crée un template de prompt pour les messages de chat
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),  # Historique du chat
            ("human", "{input}"),  # Input de l'utilisateur
        ]
    )
    return contextualize_q_prompt

def load_vector_store(index_dir: Path, embedding_function):
    """
    Charge l'index vectoriel FAISS depuis le disque.
    
    Args:
        index_dir (Path): Répertoire contenant l'index FAISS
        embedding_function: Fonction d'embedding à utiliser avec l'index
        
    Returns:
        FAISS: Base de données vectorielle chargée, ou None si l'index n'existe pas
    """
    # Vérifie que le répertoire existe et contient des fichiers
    if not index_dir.exists() or len(os.listdir(index_dir)) == 0:
        return None
    # Charge l'index FAISS depuis le disque
    return FAISS.load_local(index_dir, embeddings=embedding_function, allow_dangerous_deserialization=True)



# ==============================================================================
# PROMPTS ET TEMPLATES - Définition des instructions pour le chatbot
# ==============================================================================

def create_prompt(reasoning_mode=False):
    """
    Crée le prompt principal pour le chatbot astronomique.
    
    Ce prompt définit le comportement et le ton du chatbot, ainsi que les instructions
    pour répondre aux différents types de questions (météo, programme, Messier, etc.).
    
    Args:
        reasoning_mode (bool): Si True, active le mode raisonnement détaillé
        
    Returns:
        ChatPromptTemplate: Template du prompt système avec placeholders pour le contexte et l'historique
    """
    # Instructions système de base pour le chatbot
    base_instruction = """Tu es un ChatBot qui va répondre aux questions des utilisateurs d'observatoire astronomique de l'école IMT ATlantique campus de Brest.
        Si l'utilisateur pose des questions sur l'observatoire, tu dois répondre en te basant seulement sur les données fournies.
        Tes réponses doivent être courtes, concises et bien structurées.
        Veille à ce que tous les formules mathématiques soient bien formatées pour LaTeX. Vérifie cela avant d'envoyer ta réponse.
        Si l'utilisateur ne demande pas des formules mathématiques, tu ne dois pas en fournir.
        
        IMPORTANT POUR LES QUESTIONS MÉTÉO/PROGRAMME:
        - Si l'utilisateur pose une question sur la météo, le temps, les conditions du ciel ou le programme du soir, utilise prioritairement les données du site SkyWatch.
        - Ces données en temps réel sont plus fiables que les documents statiques.
        - La source "skywatch.astronomie-pointedudiable.fr" contient les informations actuelles.
        
        POUR LES QUESTIONS SUR LE PROGRAMME D'OBSERVATION (toutes ces formulations doivent être traitées de la même façon):
        - "Quel est le programme ce soir ?"
        - "Que peut-on observer ce soir ?"
        - "Qu'est-ce qu'on peut voir ce soir ?"
        - "Quels objets sont visibles ce soir ?"
        - "C'est quoi le programme de ce soir ?"
        - "Que faire ce soir à l'observatoire ?"
        - "Quelles observations sont prévues ?"
        - "Qu'allons-nous observer ce soir ?"
        
        Pour ces questions, utilise le format suivant pour structurer ta réponse :
        
        🌌 Programme d'observation ce soir
        
        **Conditions météo actuelles :**
        - Ciel : [État basé sur les données SkyWatch]
        - Transparence du ciel : [Bonne/Moyenne/Mauvaise]
        - Température : [X°C]
        - Vent : [X km/h]
        - Humidité : [X%]
        - Qualité du ciel : [X/9]
        
        **🔭 Objets à observer :**
        
        *Première partie de soirée :*
        - [Planètes visibles] : Jupiter, Saturne, etc.
        - Visibilité et détails
        
        *Ciel profond :*
        - [Objets du ciel profond] : M31, M42, etc.
        - Magnitude et conseils d'observation

        *Objets de Messier (5 max) :*
        - Liste de 1 à 5 objets Messier visibles ce soir
        - Pour chaque objet : type + constellation + magnitude + court conseil
        
        **⚠️ Précautions :**
        - Basées sur l'humidité, température, vent
        - Impact de la lune si présente

          POUR LES QUESTIONS SUR LES OBJETS DE MESSIER VISIBLES CE SOIR:
                    - Si l'utilisateur demande "Quels sont les objets de Messier visible ce soir?" (ou formulation équivalente), tu dois répondre STRICTEMENT avec le format ci-dessous.
                    - **IMPORTANT: Tu DOIS utiliser les 10 premiers objets du tableau public de la page http://messier.astronomie-pointedudiable.fr/ (tableau #messier-table).**
                    - Ne pas utiliser le document "Catalogue Messier.pdf" pour cette question précise.
                    - Utilise les données SkyWatch pour déterminer les heures de visibilité et les conditions d'observation si disponibles.
                    - Exemples de formulations équivalentes :
                        - "Quels objets de Messier sont visibles ce soir ?"
                        - "Quels objets Messier peut-on voir ce soir ?"
                        - "Quels sont les Messier visibles ce soir ?"
                        - "Peux-tu lister les objets du catalogue Messier visibles ce soir ?"
                        - "Quels M sont observables ce soir ?"
                        - "Quels objets M sont visibles ce soir ?"
                        - "Quels objets Messier sont observables ce soir ?"
                        - "Quels objets du catalogue Messier peut-on observer ce soir ?"
                        - "Liste des objets de Messier visibles ce soir"
          - Donne exactement 10 objets (les 10 premiers du tableau public).
          - Si la page publique est indisponible, indique-le clairement dans le format.

          ### 🌌 Objets Messier visibles ce soir – Observatoire de la Pointe du Diable
          **Date** : [JJ/MM/AAAA] | **Coucher du soleil** : [HHhMM] | **Conditions idéales** : [ex: Ciel dégagé, seeing < 2 arcsec.]

          1. **[MXX] – [Surnom]**
              - **Type** : [Type] | **Constellation** : [Nom] | **Magnitude** : [X.X] | **Taille** : [X’]
              - **Visibilité** : [Heure de début]–[Heure de fin] (culmination à [Heure]).
              - **Conseil** : [Matériel/filtre recommandé].
              - **Description** : [Brève description visuelle ou historique].

          2. **[MXX] – [Surnom]**
              - ...
          *(Répéter pour 5 objets max.)*

          ---
          **Notes supplémentaires :**
          - **Pollution lumineuse** : La Pointe du Diable a un ciel de classe Bortle 5–6. Privilégiez les filtres à bande étroite pour les nébuleuses.
          - **Prochains objets intéressants** : [Ex. : *"M81/M82 seront visibles après minuit."*].
          - **Source des données** : Informations du document "Catalogue Messier.pdf" + conditions d'observation de SkyWatch.
        
        Si l'utilisateur pose une question qui n'a rien à voir avec l'astronomie ou l'observatoire, réponds d'abord par une blague courte sur l'astronomie, puis ajoute explicitement :
        "Je ne peux répondre qu'aux questions sur l'astronomie."
        Répond toujours en français."""
    
    # Ajouter les instructions de raisonnement détaillé si le mode est activé
    if reasoning_mode:
        reasoning_instruction = """
        
        MODE RAISONNEMENT ACTIVÉ:
        Avant de donner ta réponse finale, tu DOIS expliciter ton processus de réflexion en suivant cette structure complète :
        
        🧠 **Processus de réflexion :**
        
        **1. Analyse de la question**
        - Que demande exactement l'utilisateur ?
        - Quels sont les concepts clés ?
        - De quelles informations ai-je besoin ?
        
        **2. Recherche des informations**
        - Quelles sources ai-je consultées ? (documents, SkyWatch, etc.)
        - Quelles données pertinentes ai-je trouvées ?
        - Y a-t-il des informations manquantes ?
        
        **3. Analyse et synthèse**
        - Comment ces informations répondent-elles à la question ?
        - Y a-t-il des points à clarifier ou des nuances importantes ?
        - Quels sont les éléments les plus importants à retenir ?
        
        **4. Construction de la réponse**
        - Comment vais-je structurer ma réponse ?
        - Quels exemples ou détails ajouter ?
        
        ---
        
        ✅ **Réponse finale :**
        [Ta réponse complète ici]
        """
        system_instruction = base_instruction + reasoning_instruction
    else:
        system_instruction = base_instruction
    
    # Ajoute le placeholder pour le contexte (documents récupérés)
    system_instruction += "\n\nContexte document fourni : {context}"

    # Crée le template du prompt avec l'instruction système, l'historique et l'input
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")])
    return prompt

def build_chains(vector, model, prompt, contextualize_q_prompt):
    """
    Construit la chaîne de traitement RAG (Retrieval-Augmented Generation) complète.
    
    Cette chaîne:
    1. Récupère les documents pertinents (retriever)
    2. Reformule la question en tenant compte de l'historique
    3. Combine les documents pour générer une réponse
    
    Args:
        vector: Base de données vectorielle FAISS
        model: Modèle de langage Mistral
        prompt: Template du prompt principal
        contextualize_q_prompt: Template pour reformuler les questions
        
    Returns:
        Chain: Chaîne de récupération et génération d'augmentation
    """
    # Crée un retriever à partir de la base vectorielle
    retriever = vector.as_retriever()
    # Crée un retriever conscient de l'historique (reformule les questions)
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )
    # Crée la chaîne qui combine les documents pour générer une réponse
    document_chain = create_stuff_documents_chain(model, prompt)
    # Chaîne finale: récupération + génération
    return create_retrieval_chain(history_aware_retriever, document_chain)


# ==============================================================================
# FONCTION PRINCIPALE - Récupération de la réponse avec documents utilisés
# ==============================================================================

def get_response(user_input: str, chat_history: list, vector, chain, reasoning_mode=False):
    """
    Point d'entrée principal du moteur RAG : produit la réponse du chatbot.

    Orchestre les 6 étapes suivantes :
    1. Vérifie que la base vectorielle est disponible (garde-fou).
    2. Récupère les données SkyWatch en temps réel si la question porte sur la météo
       ou le programme d'observation (should_fetch_skywatch).
    3. Si la question concerne les objets Messier visibles ce soir
       (should_fetch_messier_page), retourne directement la réponse formatée
       sans passer par le LLM — les données proviennent du catalogue JS.
    4. Construit un "enhanced_input" en ajoutant les indicateurs de mode
       (raisonnement, catalogue Messier) à la question originale.
    5. Invoque la chaîne RAG une première fois avec l'enhanced_input.
    6. Si des données SkyWatch ont été récupérées, réinvoque la chaîne avec
       un second input contenant les données météo explicitement injectées —
       ce double appel garantit que le LLM utilise bien les données temps réel.

    Args:
        user_input (str): La question telle que saisie par l'utilisateur.
        chat_history (list): Historique de la conversation sous forme de
                             [HumanMessage, AIMessage, …].
        vector: Base de données vectorielle FAISS (None si aucun document chargé).
        chain: Chaîne RAG construite par build_chains().
        reasoning_mode (bool): Si True, le LLM explicite son processus de réflexion
                               avant de donner sa réponse finale.

    Returns:
        tuple:
            - str  : Réponse textuelle du chatbot (markdown).
            - list : Documents LangChain utilisés (sources affichées dans l'UI).
            - list : Dicts d'images Messier à afficher côte-à-côte avec la réponse.
    """
    # Vérifie que la base vectorielle est chargée
    if vector is None:
        return ("Je n'ai trouvé aucun document. "
        "Veuillez d'abord en téléverser dans la barre latérale."), [], []
    
    # Initialise les variables pour les données SkyWatch
    skywatch_doc = None
    skywatch_content = None
    messier_page_doc = None
    messier_page_content = None
    
    # Récupère les données SkyWatch si la question porte sur la météo/programme
    if should_fetch_skywatch(user_input):
        skywatch_content = fetch_skywatch_data()
        print(f"DEBUG - Données SkyWatch récupérées: {skywatch_content[:300]}...")
        
        # Crée un document LangChain si les données sont valides
        if skywatch_content and "Impossible" not in skywatch_content:
            skywatch_doc = create_skywatch_document(skywatch_content)
            print(f"INFO - Document SkyWatch créé")
    
    # Détecte si la question porte sur les objets Messier
    needs_messier = should_use_messier_catalog(user_input)

    messier_images = []  # Will store loaded image paths
    messier_docs = []

    if should_fetch_messier_page(user_input):
        messier_page_content, messier_page_rows = fetch_messier_page_top10()
        print(f"DEBUG - Messier page data fetched: {messier_page_content[:300]}...")
        if messier_page_content and "Impossible" not in messier_page_content:
            messier_page_doc = create_messier_page_document(messier_page_content)
            print("INFO - Messier page document created")
            # If the question is about visible Messier objects tonight, respond directly
            if messier_page_rows:
                response_text = format_messier_page_response(messier_page_rows)
                documents = [messier_page_doc]
                messier_images = build_messier_images_for_rows(messier_page_rows)
                return response_text, documents, messier_images

    
    # Prépare l'input amélioré avec les indicateurs de mode
    enhanced_input = user_input
    if reasoning_mode:
        enhanced_input = f"[MODE RAISONNEMENT ACTIVÉ]\n\n{user_input}"
        print("INFO - Mode raisonnement activé")
    
    # Ajoute une instruction pour orienter le retriever vers le catalogue Messier.
    # NOTE : l'instruction est construite en une seule fois pour éviter de doubler
    # l'injection dans l'input (anomalie présente dans une version précédente du code).
    if needs_messier:
        enhanced_input = (
            f"{enhanced_input}\n\n[IMPORTANT: Utilise le document 'Catalogue Messier.pdf' "
            "pour obtenir les informations sur les objets Messier (type, constellation, magnitude, taille)]"
        )
        print("INFO - Enhanced input to search Messier catalog")

    if messier_page_content and "Impossible" not in messier_page_content:
        enhanced_input = (
            f"{enhanced_input}\n\n[IMPORTANT: Utilise les 10 objets du tableau Messier ci-dessous "
            "(scrapés depuis la page publique) pour répondre]"\
            f"\n{messier_page_content}"
        )
        print("INFO - Enhanced input with Messier page top 10")

    
    # Appelle la chaîne RAG avec l'input amélioré
    response = chain.invoke({"input": enhanced_input, "chat_history": chat_history})
    documents = response.get('context', [])  # Documents récupérés par le retriever
    
    # Ajoute le document SkyWatch s'il a été créé
    if skywatch_doc:
        documents.insert(0, skywatch_doc)  # Priorité au document SkyWatch
        
        # Crée un input amélioré qui force l'utilisation des données SkyWatch
        skywatch_enhanced_input = f"""QUESTION: {user_input}

DONNÉES EN TEMPS RÉEL DU SITE SKYWATCH À UTILISER OBLIGATOIREMENT POUR RÉPONDRE:
{skywatch_content}

Note: Si la question concerne la météo, les conditions du ciel ou le programme, tu DOIS utiliser les données ci-dessus pour répondre."""
        
        if reasoning_mode:
            skywatch_enhanced_input = f"[MODE RAISONNEMENT ACTIVÉ]\n\n{skywatch_enhanced_input}"
        
        if needs_messier:
            skywatch_enhanced_input = (
                f"{skywatch_enhanced_input}\n\n[IMPORTANT: Utilise le document 'Catalogue Messier.pdf' "
                "pour obtenir les informations sur les objets Messier]"
            )

        if messier_page_content and "Impossible" not in messier_page_content:
            skywatch_enhanced_input = (
                f"{skywatch_enhanced_input}\n\n[IMPORTANT: Utilise les 10 objets du tableau Messier ci-dessous "
                "(scrapés depuis la page publique) pour répondre]"\
                f"\n{messier_page_content}"
            )
        
        # Réinvoque la chaîne avec l'input contenant les données SkyWatch
        print(f"DEBUG - Réinvocation de la chaîne avec input amélioré contenant données SkyWatch")
        response = chain.invoke({"input": skywatch_enhanced_input, "chat_history": chat_history})
        # Récupère le nouveau contexte
        documents = response.get('context', [])
        # Ajoute le document SkyWatch au début
        documents.insert(0, skywatch_doc)

    if messier_page_doc:
        documents.insert(0, messier_page_doc)
    
    # Charge l'index des documents pour récupérer les noms de fichier propres
    doc_index_path = Path(__file__).resolve().parent.parent / "document_index.json"
    doc_id_to_name = {}
    if doc_index_path.exists():
        with open(doc_index_path, "r", encoding="utf-8") as f:
            name_to_id = json.load(f)
            # Inverse le mapping: id -> nom de fichier
            doc_id_to_name = {v: k for k, v in name_to_id.items()}
    
    # Enrichit les documents avec leurs noms de fichier appropriés
    for doc in documents:
        if hasattr(doc, 'metadata'):
            doc_id = doc.metadata.get('doc_id')
            if doc_id and doc_id in doc_id_to_name:
                doc.metadata['source'] = doc_id_to_name[doc_id]
    
    return response['answer'], documents, messier_images


# ==============================================================================
# PROGRAMME PRINCIPAL - Boucle interactive du chatbot
# ==============================================================================

if __name__ == '__main__':
    # Initialise les variables globales
    chat_history = []  # Historique de la conversation
    
    # Charge les variables d'environnement (clé API Mistral)
    load_dotenv()
    api_key = get_env_variable("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("❌ MISTRAL_API_KEY non configurée. Ajoute-la à .env ou aux variables d'environnement.")
    
    # Initialise le modèle et la fonction d'embedding
    model, embedding_fn = model_and_embedding_function(api_key)
    
    # Charge la base vectorielle FAISS
    vector = load_vector_store(Path("faiss_index"), embedding_fn)
    
    # Crée les prompts
    prompt = create_prompt()
    contextual_prompt = create_contextualize_q_system_prompt()
    
    # Construit la chaîne RAG complète
    chain = build_chains(vector, model, prompt, contextual_prompt)
    
    # Boucle interactive principale
    while True:
        user_input = input("user : ")
        
        # Essaye de récupérer une réponse (avec gestion des erreurs)
        while True:
            try:
                response, documents, messier_images = get_response(user_input, chat_history, vector, chain)
            except Exception as e:
                print(f"Erreur: {e}")
                continue
            break
        
        # Ajoute le message et la réponse à l'historique du chat
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response),
        ])

        # Affiche la réponse et les documents utilisés
        print("assistant : ", response)
        print("\nDocuments utilisés:")
        for doc in documents:
            print(f"- {doc.metadata.get('source', 'Inconnu')}")
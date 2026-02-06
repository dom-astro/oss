# ==============================================================================
# IMPORTS - Dépendances pour le chatbot astronomique avec RAG (Retrieval-Augmented Generation)
# ==============================================================================

# LangChain - Framework pour construire des applications basées sur les LLM
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

# ==============================================================================
# CACHE GLOBAL - Stockage des données météo/programme du site SkyWatch
# ==============================================================================
# Ce cache évite de récupérer les données trop fréquemment du site SkyWatch
SKYWATCH_CACHE = {
    'data': None,  # Contient les données météo et astronomiques brutes
    'timestamp': None,  # Timestamp de la dernière récupération
    'refresh_interval': 300  # Intervalle de rafraîchissement en secondes (5 minutes)
}

# Cache pour le catalogue Messier (page publique)
MESSIER_PAGE_CACHE = {
    'data': None,
    'timestamp': None,
    'refresh_interval': 300  # 5 minutes en secondes
}

# Cache pour le catalogue Messier (page publique)
MESSIER_PAGE_CACHE = {
    'data': None,
    'timestamp': None,
    'refresh_interval': 300  # 5 minutes en secondes
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
    """Check if the question is about visible Messier objects tonight."""
    user_lower = user_input.lower()
    return should_use_messier_catalog(user_input) and (
        "visible" in user_lower or "visibles" in user_lower or "ce soir" in user_lower
    )

def get_messier_context(vector, doc_id: str, max_chunks: int = 110):
    """Retrieve relevant chunks from Catalogue Messier by doc_id (increased significantly for all objects)."""
    if vector is None or not doc_id:
        return []

    try:
        # Scan docstore for chunks of the Messier document - get as many as needed
        docs = []
        for ds_id in vector.index_to_docstore_id.values():
            doc = vector.docstore.search(ds_id)
            if getattr(doc, "metadata", {}).get("doc_id") == doc_id:
                docs.append(doc)
                if len(docs) >= max_chunks:  # Limit to 110 to cover all Messier objects
                    break
        
        if docs:
            print(f"INFO - Retrieved {len(docs)} chunks from Catalogue Messier")
            return docs
        
        # Fallback: try similarity search
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

def fetch_messier_page_top10() -> str:
    """Fetch the public Messier page and extract the 10 objects displayed in the table."""
    global MESSIER_PAGE_CACHE

    now = datetime.now()
    if MESSIER_PAGE_CACHE['data'] is not None and MESSIER_PAGE_CACHE['timestamp'] is not None:
        if (now - MESSIER_PAGE_CACHE['timestamp']).total_seconds() < MESSIER_PAGE_CACHE['refresh_interval']:
            return MESSIER_PAGE_CACHE['data']

    url = "http://messier.astronomie-pointedudiable.fr/"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.select_one("#messier-table")
        if not table:
            return "Impossible de récupérer le tableau Messier (tableau introuvable)."

        rows = table.select("tbody tr")
        if not rows:
            return "Impossible de récupérer les objets Messier (aucune ligne trouvée dans le tableau)."

        objects = []
        for row in rows[:10]:
            cells = [cell.get_text(strip=True) for cell in row.find_all("td")]
            if len(cells) < 7:
                continue
            objects.append({
                "messier": cells[0],
                "objet": cells[2],
                "saison": cells[3],
                "mag": cells[4],
                "constellation": cells[5],
                "visible": cells[6]
            })

        if not objects:
            return "Impossible de récupérer les objets Messier (lignes incomplètes)."

        lines = [
            "LISTE DES 10 OBJETS MESSIER AFFICHÉS (source: http://messier.astronomie-pointedudiable.fr/)",
        ]
        for idx, obj in enumerate(objects, 1):
            lines.append(
                f"{idx}. {obj['messier']} | {obj['objet']} | Saison: {obj['saison']} | Mag: {obj['mag']} | Constellation: {obj['constellation']} | Visible: {obj['visible']}"
            )

        content = "\n".join(lines)
        MESSIER_PAGE_CACHE['data'] = content
        MESSIER_PAGE_CACHE['timestamp'] = now
        return content
    except Exception as e:
        return f"Impossible de récupérer la page Messier: {e}"

def create_messier_page_document(messier_page_content: str):
    """Create a LangChain Document from Messier page data"""
    from langchain_core.documents import Document
    return Document(
        page_content=messier_page_content,
        metadata={
            'source': 'messier.astronomie-pointedudiable.fr',
            'type': 'messier_page_top10'
        }
    )

def find_messier_info(messier_number: int, messier_docs: list) -> str:
    """Find comprehensive info snippet for a given Messier number - search across all chunks."""
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
    """Extract Messier numbers from text in order of appearance - handles multiple formats."""
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
            'type': 'realtime_weather'  # Type de données: météo en temps réel
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
                    - **IMPORTANT: Tu DOIS utiliser les informations du document "Catalogue Messier.pdf" disponible dans le contexte pour identifier les objets, leurs caractéristiques (type, constellation, magnitude, taille).**
                    - Utilise les données SkyWatch pour déterminer les heures de visibilité et les conditions d'observation.
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
          - Ne dépasse pas 5 objets.
          - Si les informations ne sont pas disponibles dans le Catalogue Messier.pdf, indique-le clairement dans le format.

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
        
        Si l'utilisateur pose des questions sur quelque chose autre que l'observatoire, tu refuses de répondre.
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
    Récupère la réponse du chatbot pour une question utilisateur.
    
    Cette fonction gère:
    - La récupération des données SkyWatch si nécessaire
    - La détection des questions sur les objets Messier
    - L'invocation de la chaîne RAG
    - Le retour des documents utilisés
    
    Args:
        user_input (str): La question de l'utilisateur
        chat_history (list): Historique de la conversation
        vector: Base de données vectorielle FAISS
        chain: Chaîne RAG construite
        reasoning_mode (bool): Activer le mode raisonnement détaillé
        
    Returns:
        tuple: (réponse_texte, documents_utilisés)
    """
    # Vérifie que la base vectorielle est chargée
    if vector is None:
        return ("Je n'ai trouvé aucun document. "
        "Veuillez d'abord en téléverser dans la barre latérale."), []
    
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
<<<<<<< Updated upstream
=======
    messier_images = []  # Will store loaded image paths
    messier_docs = []

    if should_fetch_messier_page(user_input):
        messier_page_content = fetch_messier_page_top10()
        print(f"DEBUG - Messier page data fetched: {messier_page_content[:300]}...")
        if messier_page_content and "Impossible" not in messier_page_content:
            messier_page_doc = create_messier_page_document(messier_page_content)
            print("INFO - Messier page document created")
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
    
    # Prépare l'input amélioré avec les indicateurs de mode
    enhanced_input = user_input
    if reasoning_mode:
        enhanced_input = f"[MODE RAISONNEMENT ACTIVÉ]\n\n{user_input}"
        print("INFO - Mode raisonnement activé")
    
    # Ajoute une instruction pour rechercher le catalogue Messier si nécessaire
    if needs_messier:
<<<<<<< Updated upstream
        enhanced_input = f"{enhanced_input}\n\n[IMPORTANT: Rechercher dans le document 'Catalogue Messier.pdf' pour obtenir les informations sur les objets Messier (type, constellation, magnitude, taille)]"
        print("INFO - Input amélioré pour recherche dans catalogue Messier")
=======
        # Only inject actual Messier context without doubling it
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
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
    
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
    
    return response['answer'], documents


# ==============================================================================
# PROGRAMME PRINCIPAL - Boucle interactive du chatbot
# ==============================================================================

if __name__ == '__main__':
    # Initialise les variables globales
    chat_history = []  # Historique de la conversation
    
    # Charge les variables d'environnement (clé API Mistral)
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    
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
                response, documents = get_response(user_input, chat_history, vector, chain)
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
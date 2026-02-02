from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage ,AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from dotenv import load_dotenv
from pathlib import Path
import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re

# Cache pour les données SkyWatch
SKYWATCH_CACHE = {
    'data': None,
    'timestamp': None,
    'refresh_interval': 300  # 5 minutes en secondes
}

def model_and_embedding_function(api_key):
    # Create embedding function using Mistral AI's embedding model
    embedding_function = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)

    # Initialize the language model with Mistral AI
    model = ChatMistralAI(mistral_api_key=api_key, model="mistral-large-latest")
    return model, embedding_function

def should_fetch_skywatch(user_input: str) -> bool:
    """Check if the question is about weather or tonight's program"""
    keywords = ['météo', 'meteo', 'temps', 'soir', 'programme', 'ce soir', 'pluie', 'nuages', 'ciel', 'conditions', 'beau', 'observation', 'sky', 'weather', 'sky watch', 'skywatch', 'nuit', 'seeing', 'transparence', 'couverture', 'couverture nuageuse', 'observer', 'visible', 'visibilité']
    user_lower = user_input.lower()
    
    # Check if any keyword is in the input
    should_fetch = any(keyword in user_lower for keyword in keywords)
    
    if should_fetch:
        print(f"INFO - SkyWatch fetch triggered for: {user_input}")
    
    return should_fetch

def should_use_messier_catalog(user_input: str) -> bool:
    """Check if the question is about Messier objects"""
    keywords = ['messier', 'catalogue messier', 'objets messier', 'objets de messier', 'objet messier', 'objet de messier', ' m31', ' m42', ' m45', ' m13', ' m1 ', 'objets m ']
    user_lower = user_input.lower()
    
    # Check if any keyword is in the input
    should_use = any(keyword in user_lower for keyword in keywords)
    
    if should_use:
        print(f"INFO - Messier catalog usage triggered for: {user_input}")
    
    return should_use

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

def find_messier_info(messier_number: int, messier_docs: list) -> str:
    """Find comprehensive info snippet for a given Messier number - search across all chunks."""
    if not messier_number or not messier_docs:
        return ""

    # Build a comprehensive list of patterns to search for
    patterns = [
        f"M {messier_number}",          # M 31
        f"M{messier_number}",           # M31
        f"M -{messier_number}",         # M -31
        f"M-{messier_number:03d}",      # M-031
        f"M {messier_number:03d}",      # M 031
        f"M{messier_number:03d}",       # M031
        f"( M {messier_number}",        # ( M 31
        f"( M{messier_number}",         # ( M31
        f"M {messier_number:02d}",      # M 31 (without leading zero)
        f"M{messier_number:02d}",       # M31 (without leading zero)
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
                if pos < best_position or (pos == best_position and len(text) > len(best_match)):
                    best_match = text
                    best_position = pos
                break
    
    # Return up to 900 chars for better information display
    return best_match[:900] if best_match else ""

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
    """Fetch weather and program data from skywatch website with 5-minute cache"""
    global SKYWATCH_CACHE
    
    # Vérifier si les données en cache sont encore valides
    now = datetime.now()
    if SKYWATCH_CACHE['data'] is not None and SKYWATCH_CACHE['timestamp'] is not None:
        age = (now - SKYWATCH_CACHE['timestamp']).total_seconds()
        if age < SKYWATCH_CACHE['refresh_interval']:
            print(f"INFO - Using cached SkyWatch data (age: {int(age)}s)")
            return SKYWATCH_CACHE['data']
        else:
            print(f"INFO - Cache expired (age: {int(age)}s), refreshing...")
    
    try:
        # The main URL redirects to this one
        urls = [
            "http://nas-gdl2.synology.me/skywatch/",
            "http://skywatch.astronomie-pointedudiable.fr/"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = None
        for url in urls:
            try:
                response = requests.get(url, timeout=10, headers=headers, allow_redirects=True)
                print(f"DEBUG - Connected to {url}, status: {response.status_code}")
                if response.status_code == 200:
                    break
            except Exception as e:
                print(f"DEBUG - Failed for {url}: {e}")
                continue
        
        if not response:
            # Si échec, retourner les données en cache si disponibles
            if SKYWATCH_CACHE['data']:
                print("WARNING - Connection failed, using old cache")
                return SKYWATCH_CACHE['data']
            return "Impossible de se connecter au site SkyWatch"
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get all text content - the site doesn't use proper h2/h3 headers
        all_text = soup.get_text(separator='\n', strip=True)
        lines = [l.strip() for l in all_text.split('\n') if l.strip()]
        
        extracted_data = []
        extracted_data.append("=== DONNÉES MÉTÉO ET ASTRONOMIQUES EN TEMPS RÉEL DU SITE SKYWATCH ===\n")
        extracted_data.append(f"Dernière mise à jour : {now.strftime('%d/%m/%Y %H:%M:%S')}\n")
        
        # Extract weather data - it appears after "Infos météo" or directly
        # Look for key weather fields
        weather_keywords = {
            'Date', 'Heure', 'Température', 'Vent', 'Humidité', 
            'Lever du soleil', 'Coucher du soleil', 'Qualité du ciel',
            'Météo', 'Description', 'Ville'
        }
        
        extracted_data.append("\n** MÉTÉO ACTUELLE **")
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check if this line is a weather keyword
            if any(keyword in line for keyword in weather_keywords):
                # The value is usually the next line
                if i + 1 < len(lines):
                    value = lines[i + 1]
                    # Make sure the value isn't another keyword
                    if not any(kw in value for kw in weather_keywords) and len(value) < 50:
                        extracted_data.append(f"{line}: {value}")
                        i += 2  # Skip both lines
                        continue
            i += 1
        
        # Extract planet ephemeris data
        extracted_data.append("\n** ÉPHÉMÉRIDES DES PLANÈTES **")
        planet_names = ['Mercure', 'Vénus', 'Mars', 'Jupiter', 'Saturne', 'Uranus', 'Neptune']
        
        for planet in planet_names:
            if planet in lines:
                idx = lines.index(planet)
                extracted_data.append(f"{planet}:")
                # Get some data after the planet name (lever time, etc.)
                for j in range(idx + 1, min(idx + 5, len(lines))):
                    if lines[j] and len(lines[j]) < 30 and not any(p in lines[j] for p in planet_names):
                        extracted_data.append(f"  {lines[j]}")
        
        # If we got good data, cache it
        if len(extracted_data) > 5:
            result = '\n'.join(extracted_data[:50])
            # Mettre à jour le cache
            SKYWATCH_CACHE['data'] = result
            SKYWATCH_CACHE['timestamp'] = now
            print(f"INFO - SkyWatch data cached ({len(extracted_data)} fields)")
            return result
        
        # Fallback: just return the relevant portion of text
        print("DEBUG - Using fallback text extraction")
        # Find the section with weather data (usually after "Skywatch" title)
        start_idx = 0
        for i, line in enumerate(lines):
            if 'Date' in line or 'Heure' in line:
                start_idx = max(0, i - 5)
                break
        
        weather_section = lines[start_idx:start_idx + 60]
        result = '\n'.join(['=== DONNÉES SKYWATCH ==='] + weather_section)
        
        # Mettre à jour le cache même pour le fallback
        SKYWATCH_CACHE['data'] = result
        SKYWATCH_CACHE['timestamp'] = now
        
        return result
            
    except Exception as e:
        print(f"ERROR - Erreur lors du scraping: {str(e)}")
        # En cas d'erreur, retourner les données en cache si disponibles
        if SKYWATCH_CACHE['data']:
            print("WARNING - Error occurred, using old cache")
            return SKYWATCH_CACHE['data']
        import traceback
        traceback.print_exc()
        return f"Impossible de récupérer les données du site: {str(e)}"

def create_skywatch_document(skywatch_content: str):
    """Create a LangChain Document from skywatch data"""
    from langchain_core.documents import Document
    return Document(
        page_content=skywatch_content,
        metadata={
            'source': 'skywatch.astronomie-pointedudiable.fr',
            'type': 'realtime_weather'
        }
    )
def create_contextualize_q_system_prompt():
    # System prompt for contextualizing questions based on chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # Create a chat prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),  # Placeholder for chat history
            ("human", "{input}"),  # Placeholder for user input
        ]
    )
    return contextualize_q_prompt

def load_vector_store(index_dir: Path,embedding_function):
    if not index_dir.exists() or len(os.listdir(index_dir)) == 0 :
        return None
    return FAISS.load_local(index_dir, embeddings=embedding_function,allow_dangerous_deserialization=True)




# Define prompt template function

def create_prompt(reasoning_mode=False):
    """
    Returns a prompt instructed to produce a rephrased question based on the user's
    last question, but referencing previous messages (chat history).
    
    Args:
        reasoning_mode: If True, includes detailed reasoning steps in the response
    """
    # System instruction in French for the astronomy observatory chatbot
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
    
    # Ajouter les instructions de raisonnement si le mode est activé
    if reasoning_mode:
        reasoning_instruction = """
        
        MODE RAISONNEMENT ACTIVÉ:
        Avant de donner ta réponse finale, tu DOIS expliciter ton processus de réflexion en suivant cette structure :
        
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
    
    system_instruction += "\n\nUtiliser le context : {context}"

    # Create chat prompt template with system instruction, chat history and user input
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")])
    return prompt

def build_chains(vector, model, prompt, contextualize_q_prompt):
    retriever  = vector.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )
    document_chain = create_stuff_documents_chain(model, prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)



# Modifier get_response pour retourner aussi les documents
def get_response(user_input: str, chat_history: list, vector, chain, reasoning_mode=False):
    if vector is None:
        return ("Je n'ai trouvé aucun document. "
        "Veuillez d'abord en téléverser dans la barre latérale."), [], []

    # Load document index to get proper filenames
    doc_index_path = Path(__file__).resolve().parent.parent / "document_index.json"
    doc_id_to_name = {}
    name_to_id = {}
    if doc_index_path.exists():
        with open(doc_index_path, "r", encoding="utf-8") as f:
            name_to_id = json.load(f)
            doc_id_to_name = {v: k for k, v in name_to_id.items()}
    
    # Check if we need to fetch skywatch data BEFORE calling chain
    skywatch_doc = None
    skywatch_content = None
    
    if should_fetch_skywatch(user_input):
        skywatch_content = fetch_skywatch_data()
        print(f"DEBUG - SkyWatch data fetched: {skywatch_content[:300]}...")
        
        if skywatch_content and "Impossible" not in skywatch_content:
            skywatch_doc = create_skywatch_document(skywatch_content)
            print(f"INFO - SkyWatch document created")
    
    # Check if we need to explicitly search for Messier catalog
    needs_messier = should_use_messier_catalog(user_input)
    messier_images = []  # Will store loaded image paths
    messier_docs = []
    
    if needs_messier:
        # Load images from assets instead of extracting from PDF
        print(f"INFO - Loading Messier images from assets")
        messier_images = load_messier_images_from_assets(max_images=5)
        messier_doc_id = name_to_id.get("Catalogue Messier.pdf")
        if messier_doc_id:
            messier_docs = get_messier_context(vector, messier_doc_id, max_chunks=5)
            if messier_docs:
                print(f"INFO - Loaded {len(messier_docs)} Messier context chunks")
        # Attach info snippets to images
        if messier_docs and messier_images:
            for img in messier_images:
                img["info"] = find_messier_info(img.get("messier_number"), messier_docs)
    
    # Add reasoning mode indicator to input if active
    enhanced_input = user_input
    if reasoning_mode:
        enhanced_input = f"[MODE RAISONNEMENT ACTIVÉ]\n\n{user_input}"
        print("INFO - Reasoning mode activated")
    
    # If Messier objects are mentioned, enhance the query to include catalog search
    if needs_messier:
        # Only inject actual Messier context without doubling it
        enhanced_input = (
            f"{enhanced_input}\n\n[IMPORTANT: Utilise le document 'Catalogue Messier.pdf' "
            "pour obtenir les informations sur les objets Messier (type, constellation, magnitude, taille)]"
        )
        print("INFO - Enhanced input to search Messier catalog")
    
    # Invoke the chain with original input
    response = chain.invoke({"input": enhanced_input, "chat_history": chat_history})
    documents = response.get('context', [])  # Les documents récupérés
    
    # Add skywatch doc to documents list - it will be included in the context
    if skywatch_doc:
        documents.insert(0, skywatch_doc)
        
        # Create a modified prompt that forces using skywatch data
        # Re-run with enhanced input that includes skywatch info
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
        
        # Re-invoke with enhanced input
        print(f"DEBUG - Re-invoking chain with enhanced input")
        response = chain.invoke({"input": skywatch_enhanced_input, "chat_history": chat_history})
        # Get the new context
        documents = response.get('context', [])
        # Add skywatch doc back to beginning
        documents.insert(0, skywatch_doc)
    
    # Add Messier docs to sources if available
    if messier_docs:
        documents = messier_docs + documents
    
    # Enrich documents with proper filenames
    for doc in documents:
        if hasattr(doc, 'metadata'):
            doc_id = doc.metadata.get('doc_id')
            if doc_id and doc_id in doc_id_to_name:
                doc.metadata['source'] = doc_id_to_name[doc_id]

    # Filter Messier images to match the objects mentioned in the answer
    if needs_messier and messier_images:
        mentioned_numbers = extract_messier_numbers(response.get('answer', ''))
        print(f"DEBUG - Full answer text: {response.get('answer', '')[:500]}")
        print(f"DEBUG - Mentioned Messier numbers in answer: {mentioned_numbers}")
        print(f"DEBUG - Available image numbers: {[img.get('messier_number') for img in messier_images[:20]]}")
        
        if mentioned_numbers and len(mentioned_numbers) > 0:
            ordered_images = []
            for num in mentioned_numbers[:5]:  # Only take first 5 mentioned
                found = False
                for img in messier_images:
                    if img.get('messier_number') == num:
                        ordered_images.append(img)
                        print(f"DEBUG - ✓ Matched M{num} from answer with image")
                        found = True
                        break
                if not found:
                    print(f"DEBUG - ✗ M{num} mentioned but image not found")
            
            messier_images = ordered_images
            print(f"DEBUG - Final selected {len(messier_images)} images from {len(mentioned_numbers)} mentioned objects")
            print(f"DEBUG - Final selected image numbers: {[img.get('messier_number') for img in messier_images]}")
        else:
            print(f"DEBUG - No Messier numbers found in answer - clearing images")
            messier_images = []

    return response['answer'], documents, messier_images

if __name__ == '__main__' :
    chat_history = []
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    model, embedding_fn = model_and_embedding_function(api_key)
    vector = load_vector_store(Path("faiss_index"), embedding_fn)
    prompt = create_prompt()
    contextual_prompt = create_contextualize_q_system_prompt()
    chain = build_chains(vector, model, prompt, contextual_prompt)
    while True:
        user_input = input("user : ")
        while True:
            try:
                response, documents, messier_images = get_response(user_input,chat_history,vector,chain)
            except Exception as e :
                print(e)
                continue
            break
        chat_history.extend(
        [
            HumanMessage(content=user_input),
            AIMessage(content=response),
        ]
        )

        print("assisatnt : ",response)
        print("\nDocuments utilisés:")
        for doc in documents:
            print(f"- {doc.metadata.get('source', 'Unknown')}")
        if messier_images:
            print(f"\nImages Messier trouvées: {len(messier_images)}")
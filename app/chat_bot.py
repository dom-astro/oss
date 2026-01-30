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
        
        **⚠️ Précautions :**
        - Basées sur l'humidité, température, vent
        - Impact de la lune si présente
        
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
        "Veuillez d'abord en téléverser dans la barre latérale."), []
    
    # Check if we need to fetch skywatch data BEFORE calling chain
    skywatch_doc = None
    skywatch_content = None
    
    if should_fetch_skywatch(user_input):
        skywatch_content = fetch_skywatch_data()
        print(f"DEBUG - SkyWatch data fetched: {skywatch_content[:300]}...")
        
        if skywatch_content and "Impossible" not in skywatch_content:
            skywatch_doc = create_skywatch_document(skywatch_content)
            print(f"INFO - SkyWatch document created")
    
    # Add reasoning mode indicator to input if active
    enhanced_input = user_input
    if reasoning_mode:
        enhanced_input = f"[MODE RAISONNEMENT ACTIVÉ]\n\n{user_input}"
        print("INFO - Reasoning mode activated")
    
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
        
        # Re-invoke with enhanced input
        print(f"DEBUG - Re-invoking chain with enhanced input")
        response = chain.invoke({"input": skywatch_enhanced_input, "chat_history": chat_history})
        # Get the new context
        documents = response.get('context', [])
        # Add skywatch doc back to beginning
        documents.insert(0, skywatch_doc)
    
    # Load document index to get proper filenames
    doc_index_path = Path(__file__).resolve().parent.parent / "document_index.json"
    doc_id_to_name = {}
    if doc_index_path.exists():
        with open(doc_index_path, "r", encoding="utf-8") as f:
            name_to_id = json.load(f)
            # Reverse the mapping: id -> name
            doc_id_to_name = {v: k for k, v in name_to_id.items()}
    
    # Enrich documents with proper filenames
    for doc in documents:
        if hasattr(doc, 'metadata'):
            doc_id = doc.metadata.get('doc_id')
            if doc_id and doc_id in doc_id_to_name:
                doc.metadata['source'] = doc_id_to_name[doc_id]
    
    return response['answer'], documents

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
                response, documents = get_response(user_input,chat_history,vector,chain)
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
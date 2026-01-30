import streamlit as st, yaml, streamlit_authenticator as stauth
from yaml import SafeLoader
from chat_bot import load_vector_store, get_response, build_chains, model_and_embedding_function, create_prompt, create_contextualize_q_system_prompt
from Embedder import Embedder
from TextEmbedder import TextEmbedder
from EmbedderWithOcr import EmbedderWithOcr
from MultimodalEmbedder import MultimodalEmbedder
from langchain_core.messages import HumanMessage, AIMessage
import smtplib
from email.mime.text import MIMEText
from email.mime.text import MIMEText
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import torch, types
torch.classes.__path__ = types.SimpleNamespace(_path=[])


# Page configuration
st.set_page_config(
    page_title="Chatbot Observatoire Astronomique", 
    page_icon=":astronaut:", 
    layout="wide",
    menu_items={
        'About': "Observatoire Astronomique - IMT Atlantique, campus de Brest"
    }
)

load_dotenv(override=True)


def send_email(receiver_email, subject, body):
    smtp_host = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")
    server = smtplib.SMTP(smtp_host, smtp_port)

    server.connect(smtp_host, smtp_port)
    server.starttls()
    server.login(smtp_user, smtp_pass)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = receiver_email

    server.send_message(msg)
    server.quit()



# Load environment variables
api_key = os.getenv("MISTRAL_API_KEY")

config_path = Path(__file__).resolve().parent.parent / "config.yaml"

# --- load config dynamically so new users persist -------------
authenticator = stauth.Authenticate(str(config_path))
# ------------- Register new user panel ------------------------

with st.sidebar:
    st.subheader("Gestion des utilisateurs")
    with st.expander("Options de gestion de compte", expanded=True):
        action = st.radio(
            "Sélectionnez une action",
            ["Se connecter", "Créer un compte", "Changer le mot de passe", "Modifier mes informations", "Mot de passe oublié", "Nom d'utilisateur oublié", "Changer le rôle d'un utilisateur"],
            index=0
        )

        if action == "Se connecter":
            try:
                authenticator.login()
            except Exception as e:
                st.sidebar.error(e)

        elif action == "Créer un compte":
            try:
                email, username, name = authenticator.register_user(password_hint=False, roles=["utilisateur"])
                if email and username and name:
                    st.sidebar.success(f"Utilisateur `{username}` enregistré avec succès !")
            except Exception as e:
                st.sidebar.error(e)

        elif action == "Changer le mot de passe":
            if st.session_state.get('authentication_status'):
                try:
                    if authenticator.reset_password(st.session_state.get('username')):
                        st.sidebar.success("Mot de passe modifié avec succès")
                except Exception as e:
                    st.sidebar.error(e)
            else:
                st.sidebar.info("Vous devez être connecté pour modifier le mot de passe.")

        elif action == "Modifier mes informations":
            if st.session_state.get('authentication_status'):
                try:
                    if authenticator.update_user_details(st.session_state.get('username')):
                        st.sidebar.success("Informations mises à jour avec succès")
                except Exception as e:
                    st.sidebar.error(e)
            else:
                st.sidebar.info("Vous devez être connecté pour modifier vos informations.")
        elif action == "Changer le rôle d'un utilisateur":
            if st.session_state.get('authentication_status'):
                user_roles = st.session_state.get("roles")
                if "admin" in user_roles:
                    if st.session_state.get('authentication_status'):
                        try:
                            with open(config_path, "r") as f:
                                config = yaml.load(f, Loader=SafeLoader)
                                user_to_change = st.selectbox("Sélectionnez un utilisateur", config["credentials"]["usernames"])
                                new_role = st.selectbox("Sélectionnez un nouveau rôle", ["admin", "utilisateur"])
                                if st.button("Changer le rôle"):
                                    config["credentials"]["usernames"][user_to_change]["roles"] = [new_role]
                                    if user_to_change and new_role:
                                        with open(config_path, "w") as f:
                                            yaml.dump(config, f)
                                        st.sidebar.success(f"Rôle de `{user_to_change}` changé en `{new_role}` avec succès !")
                                    else:
                                        st.sidebar.error("Veuillez sélectionner un utilisateur et un rôle.")
                        except Exception as e:
                            st.sidebar.error(e)
                    else:
                        st.sidebar.info("Vous devez être connecté pour modifier le rôle d'un utilisateur.")
                else:
                    st.sidebar.info("Vous devez être administrateur pour modifier le rôle d'un utilisateur.")
            else:
                st.sidebar.info("Vous devez être connecté pour modifier le rôle d'un utilisateur.")
        elif action == "Mot de passe oublié":
            try:
                username_of_forgotten_password, \
                email_of_forgotten_password, \
                new_random_password = authenticator.forgot_password()
                if username_of_forgotten_password:
                    send_email(
                        email_of_forgotten_password,
                        "Votre nouveau mot de passe",
                        f"Bonjour,\n\nVotre nouveau mot de passe est : {new_random_password} ")
                    st.success('Mot de passe oublié. Un e-mail a été envoyé à l\'adresse fournie.')
                    # To securely transfer the new password to the user please see step 8.
                elif username_of_forgotten_password == False:
                    st.error('Nom d\'utilisateur ou e-mail non trouvé.')
            except Exception as e:
                st.error(e)
        elif action == "Nom d'utilisateur oublié":
            try:
                username_of_forgotten_username, \
                email_of_forgotten_username = authenticator.forgot_username()
                if username_of_forgotten_username:
                    send_email(
                        email_of_forgotten_username,
                        "Rappel de votre identifiant",
                        f"Bonjour,\n\nVotre identifiant est : {username_of_forgotten_username}"
                    )
                    st.success('Nom d\'utilisateur oublié. Un e-mail a été envoyé à l\'adresse fournie.')
                    # To securely transfer the username to the user please see step 8.
                elif username_of_forgotten_username == False:
                    st.error('Nom d\'utilisateur ou e-mail non trouvé.')
            except Exception as e:
                st.error(e)
        
    if st.session_state.get('authentication_status'):
            authenticator.logout()


if st.session_state.get('authentication_status') is False:
    st.sidebar.error('Nom d\'utilisateur ou mot de passe incorrect. Veuillez réessayer.')
elif st.session_state.get('authentication_status') is None:
    st.sidebar.warning('Merci de vous connecter pour accéder à l\'application.')
# --- Main application logic ---
elif st.session_state.get('authentication_status'):
    if api_key is None:
        raise st.error("MISTRAL_API_KEY n'est pas défini dans les variables d'environnement")

    # Directory paths
    DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
    INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"
    HISTORY_DIR = Path(__file__).resolve().parent.parent / "chat_histories"

    # Create directories if they don't exist
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    # --- User input at the beginning of the session ---
   
    user_name = st.session_state.get("name")
    user_roles = st.session_state.get("roles")
  

    # --- Loading user history ---
    file_safe_name = user_name.lower().replace(' ', '_')
    user_history_path = HISTORY_DIR / file_safe_name
    os.makedirs(user_history_path, exist_ok=True)

    # Initialize model and vector store
    model, embedding_function = model_and_embedding_function(api_key)
    contextualize_q_prompt = create_contextualize_q_system_prompt()
    
    # Initialize reasoning mode state if not exists
    if "reasoning_mode" not in st.session_state:
        st.session_state.reasoning_mode = False
    
    # Create prompt based on reasoning mode (dynamically updated)
    prompt = create_prompt(reasoning_mode=st.session_state.get("reasoning_mode", False))

    if "vector" not in st.session_state:
        st.session_state.vector = load_vector_store(INDEX_DIR, embedding_function)
    
    # Rebuild chain if reasoning mode changed or chain doesn't exist
    rebuild_chain = False
    if "chain" not in st.session_state:
        rebuild_chain = True
    elif "last_reasoning_mode" in st.session_state:
        if st.session_state.last_reasoning_mode != st.session_state.reasoning_mode:
            rebuild_chain = True
    
    if rebuild_chain:
        if st.session_state.vector is None:
            st.session_state.chain = None
        else:
            st.session_state.chain = build_chains(
                vector=st.session_state.vector,
                model=model,
                prompt=prompt,
                contextualize_q_prompt=contextualize_q_prompt,
            )
            # Remember the reasoning mode for which this chain was built
            st.session_state.last_reasoning_mode = st.session_state.reasoning_mode

    # Initialize session state
    if "available_documents" not in st.session_state:
        st.session_state.available_documents = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "history_path" not in st.session_state:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state.history_path = user_history_path / f"{ts}.json"

    # Function to refresh available documents
    def refresh_document_list():
        if DOCS_DIR.exists():
            st.session_state.available_documents = [f for f in os.listdir(DOCS_DIR) if f.endswith(('.pdf', '.PDF'))]
        else:
            st.session_state.available_documents = []

    # Function to save chat history
    def save_chat_history():
        try:
            history_data = {
                "messages": st.session_state.messages,
                "chat_history": [
                    {"type": "human", "content": msg.content} if isinstance(msg, HumanMessage)
                    else {"type": "ai", "content": msg.content}
                    for msg in st.session_state.chat_history
                ]
            }
            
            # Convert Path to string to ensure serializability
            history_path_str = str(st.session_state.history_path)
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(history_path_str), exist_ok=True)
            
            # Save with proper encoding
            with open(history_path_str, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            st.error(f"Error saving chat history: {e}")
            return False

    # Initialize document list
    refresh_document_list()

    # --- Introduction ---
    st.markdown(f"""
    Salut **{user_name}** ! 👋  
    Je suis ton assistant virtuel pour l'observatoire astronomique. 😊  
    Je peux t'aider avec l'utilisation et la configuration des équipements
    Pose-moi une question ou demande-moi de l'aide !
    """)

    # Add a title
    st.title("Bienvenue dans l'Observatoire Astronomique 🚀")

    with st.sidebar:
        # Mode raisonnement toggle
        st.subheader("⚙️ Options")
        if "reasoning_mode" not in st.session_state:
            st.session_state.reasoning_mode = False
        
        reasoning_toggle = st.toggle(
            "🧠 Mode raisonnement détaillé",
            value=st.session_state.reasoning_mode,
            help="Affiche le processus de réflexion étape par étape pour chaque réponse"
        )
        
        if reasoning_toggle != st.session_state.reasoning_mode:
            st.session_state.reasoning_mode = reasoning_toggle
            if reasoning_toggle:
                st.success("✅ Mode raisonnement activé - Les réponses détailleront le processus de réflexion")
            else:
                st.info("Mode raisonnement désactivé - Réponses concises")
        
        st.divider()
        
        st.subheader("Historique de la discussion")
        with st.expander("Historique de la discussion", expanded=False):
            if "admin" in user_roles:
                # List all user directories
                user_directories = [d for d in os.listdir(HISTORY_DIR) if os.path.isdir(os.path.join(HISTORY_DIR, d))]
                if not user_directories:
                    st.info("Aucun utilisateur enregistré pour le moment.")
                else:
                    selected_user = st.selectbox("Choisissez un utilisateur :", user_directories, index=None, placeholder="— choisir un utilisateur —") 
                    if selected_user:
                        user_history_path = HISTORY_DIR / selected_user
            # Get list of history files for this user
            history_files = []
            if user_history_path.exists():
                history_files = sorted([f for f in os.listdir(user_history_path) if f.endswith('.json')], 
                                        reverse=True)  # Most recent first
            
            if history_files:
                selected_history = st.selectbox(
                    "Charger une conversation précédente:",
                    options=[os.path.splitext(f)[0] for f in history_files],
                    index=None,
                    placeholder="— choisir une conversation —"
                )
                
                if selected_history and st.button("Charger cette conversation"):
                    try:
                        history_file_path = user_history_path / selected_history
                        with open(history_file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            st.session_state.messages = data.get("messages", [])
                            st.session_state.chat_history = [
                                HumanMessage(content=msg["content"]) if msg["type"] == "human"
                                else AIMessage(content=msg["content"])
                                for msg in data.get("chat_history", [])
                            ]
                            st.session_state.history_path = history_file_path
                            st.success(f"Conversation chargée avec succès!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Erreur lors du chargement: {e}")
            else:
                st.info("Aucun historique disponible pour cet utilisateur.")
                
            if st.button("Nouvelle conversation"):
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                st.session_state.history_path = user_history_path / f"{ts}.json"
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success("Nouvelle conversation créée!")
                st.rerun()

        st.subheader("Gérer vos documents")
        # Document upload section
        with st.expander("Télécharger un document", expanded=False):
            pdf = st.file_uploader("Téléchargez vos documents ici et cliquez sur 'Traiter'")
            options = ["Text only embedder", "Embedder with OCR", "Multimodal embedder"]
            if pdf is not None:
                selected_embedder = st.selectbox(
                    "Choisissez l'embedder", 
                    options, 
                    index=None, 
                    placeholder="— choisir un embedder —"
                )
                
                embedder = Embedder(api_key)  # Default
                if selected_embedder == "Text only embedder":
                    embedder = TextEmbedder(api_key)
                elif selected_embedder == "Embedder with OCR":
                    embedder = EmbedderWithOcr(api_key)
                elif selected_embedder == "Multimodal embedder":
                    embedder = MultimodalEmbedder(api_key)
                add_permanently = False
                if "admin" in user_roles:
                    add_permanently = st.checkbox("Ajouter le document de manière permanente")
                if st.button("Traiter", key="process_button"):
                    try:
                        # Build destination path
                        save_path = os.path.join(DOCS_DIR, pdf.name)
                        with open(save_path, "wb") as f:
                            f.write(pdf.read())
                        if "admin" in user_roles:
                            st.success(f"Fichier enregistré dans : {save_path}")
                        with st.spinner("Indexation du document…"):
                            # Process the document
                            st.session_state.vector = embedder.embed(save_path, st.session_state.vector, save=add_permanently)
                            st.session_state.chain = build_chains(
                                vector=st.session_state.vector,
                                model=model,
                                prompt=prompt,
                                contextualize_q_prompt=contextualize_q_prompt,
                            )
                            refresh_document_list()
                        st.success("Documents traités avec succès !")
                        if "admin" in user_roles:
                            os.remove(save_path)
                    except Exception as e:
                        st.error(f"Erreur lors du traitement du document : {e}")

        # Document deletion section
        with st.expander("Supprimer un document", expanded=False):
            if st.session_state.available_documents:
                selected_doc = st.selectbox(
                    'Sélectionnez le document à supprimer',
                    st.session_state.available_documents,
                    index=None,
                    placeholder="— choisir un document —" 
                )
                delete_permanently = False
                if "admin" in user_roles:
                    delete_permanently = st.checkbox("Supprimer le document de manière permanente")
                if selected_doc and st.button("Supprimer", key="delete_btn"):
                    try:
                        with st.spinner("Suppression en cours..."):
                            file_path = os.path.join(DOCS_DIR, selected_doc)
                            delection_state = Embedder(api_key).delete_document(selected_doc, st.session_state.vector, save=delete_permanently)
                            if delection_state is False:
                                st.error(f"Le document '{selected_doc}' n'existe pas ou n'a pas pu être supprimé.")
                            else:
                                if delete_permanently:
                                    os.remove(file_path)
                                st.session_state.chain = build_chains(
                                    vector=st.session_state.vector,
                                    model=model,
                                    prompt=prompt,
                                    contextualize_q_prompt=contextualize_q_prompt,
                                )
                                refresh_document_list()
                                st.success(f"Document '{selected_doc}' supprimé avec succès!")
                    except Exception as e:
                        st.error(f"Erreur lors de la suppression: {e}")
            else:
                st.info("Aucun document disponible à supprimer.")

        # Document list section
        with st.expander("Documents disponibles", expanded=False):
            if st.session_state.available_documents:
                st.write(f"Nombre de documents: {len(st.session_state.available_documents)}")
                for doc in st.session_state.available_documents:
                    st.text(f"📄 {doc}")
            else:
                st.info("Aucun document disponible.")
            st.markdown("---")


    # Display chat history
    st.header("Conversation")

    # Boutons de questions suggérées
    st.markdown("**Questions suggérées :**")
    col1, col2, col3, col4 = st.columns(4)
    
    suggested_questions = [
        "Quel est le programme ce soir?",
        "Quelle est la météo actuelle?",
        "Comment utiliser le télescope?",
        "Quels objets peut-on observer?"
    ]
    
    question_clicked = None
    with col1:
        if st.button(suggested_questions[0], key="q1", use_container_width=True):
            question_clicked = suggested_questions[0]
    with col2:
        if st.button(suggested_questions[1], key="q2", use_container_width=True):
            question_clicked = suggested_questions[1]
    with col3:
        if st.button(suggested_questions[2], key="q3", use_container_width=True):
            question_clicked = suggested_questions[2]
    with col4:
        if st.button(suggested_questions[3], key="q4", use_container_width=True):
            question_clicked = suggested_questions[3]

    st.markdown("---")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("role") == "assistant" and message.get("sources"):
                with st.expander("📚 Documents utilisés"):
                    # Dictionary to avoid duplicates
                    unique_sources = {}
                    for item in message["sources"]:
                        source = item.get("source") or "Unknown"
                        content = item.get("content") or ""
                        if source not in unique_sources:
                            unique_sources[source] = content

                    for i, (source, content) in enumerate(unique_sources.items(), 1):
                        # Extract just the filename from the path
                        if source != 'Unknown':
                            filename = source.split('\\')[-1].split('/')[-1]
                        else:
                            filename = source
                        st.write(f"**{i}. {filename}**")
                        st.write(f"```\n{content}\n```")

    # Chatbot - Text input interface
    user_input = st.chat_input("Posez votre question ici...")

    # Si un bouton de question suggérée est cliqué, utiliser cette question
    if question_clicked:
        user_input = question_clicked

    if user_input:
        # 1) Display/store user message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 2) Create ONE placeholder for assistant response
        assistant_slot = st.empty()

        # 3) Generation loop
        while True:
            nb_tentatives = 0
            try:
                with st.spinner("Réflexion en cours..."):
                    # Utiliser le mode raisonnement si activé
                    reasoning_mode = st.session_state.get("reasoning_mode", False)
                    response, documents = get_response(
                        user_input, 
                        st.session_state["chat_history"], 
                        st.session_state.vector, 
                        st.session_state.chain,
                        reasoning_mode=reasoning_mode
                    )
                break  # Exit if all goes well
            except Exception as e:
                nb_tentatives += 1
                # Rewrite in the same container → old text is replaced
                with assistant_slot.chat_message("assistant"):
                    st.write(f"Une erreur est survenue : {e}. Nouvelle tentative…(Tentative numéro : {nb_tentatives})")

        # 4) Final response: overwrite placeholder with real content
        with assistant_slot.chat_message("assistant"):
            st.write(response)
            # Display source documents
            sources_payload = []
            if documents:
                with st.expander("📚 Documents utilisés"):
                    # Dictionary to avoid duplicates
                    unique_sources = {}
                    for doc in documents:
                        # Try different ways to get the source
                        source = None
                        if hasattr(doc, 'metadata'):
                            source = doc.metadata.get('source')
                            if not source:
                                # Try other possible keys
                                source = doc.metadata.get('file') or doc.metadata.get('filename') or doc.metadata.get('path')
                        
                        if not source:
                            source = 'Unknown'
                        
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        sources_payload.append({"source": source, "content": content})
                        
                        if source not in unique_sources:
                            unique_sources[source] = content
                    
                    for i, (source, content) in enumerate(unique_sources.items(), 1):
                        # Extract just the filename from the path
                        if source != 'Unknown':
                            filename = source.split('\\')[-1].split('/')[-1]
                        else:
                            filename = source
                        st.write(f"**{i}. {filename}**")
                        st.write(f"```\n{content}\n```")

        # 5) History for RAG and for the page
        st.session_state["chat_history"].extend(
            [HumanMessage(content=user_input), AIMessage(content=response)]
        )
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources_payload})

        # Save chat history after each exchange
        save_chat_history()


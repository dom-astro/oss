# ==============================================================================
# IMPORTS - Dépendances pour l'interface Streamlit du chatbot
# ==============================================================================

# Framework Streamlit pour construire l'interface web
import streamlit as st  # Framework principal
import yaml  # Manipulation des fichiers de configuration YAML
import streamlit_authenticator as stauth  # Authentification des utilisateurs
from yaml import SafeLoader  # Chargement sécurisé des fichiers YAML

# Imports du module chatbot (RAG, modèles, chaînes)
from chat_bot import load_vector_store, get_response, build_chains, model_and_embedding_function, create_prompt, create_contextualize_q_system_prompt

# Embedders - Différentes méthodes d'intégration de documents
from Embedder import Embedder  # Embedder par défaut
from TextEmbedder import TextEmbedder  # Embedder texte uniquement
from EmbedderWithOcr import EmbedderWithOcr  # Embedder avec OCR pour images
from MultimodalEmbedder import MultimodalEmbedder  # Embedder multimodal

# LangChain - Messages du chat
from langchain_core.messages import HumanMessage, AIMessage  # Types de messages

# Email - Envoi de messages email
import smtplib  # Protocole SMTP pour envoi d'emails
from email.mime.text import MIMEText  # Format texte pour emails

# Utilitaires Python
import os  # Gestion des fichiers/répertoires
from env_utils import get_env_variable, detect_environment  # Gestion centralisée des variables d'environnement
import json  # Manipulation de JSON
from datetime import datetime  # Gestion des dates/heures
from dotenv import load_dotenv  # Chargement des variables d'environnement
from pathlib import Path  # Gestion des chemins fichiers

# PyTorch - Contournement d'un warning Streamlit causé par l'inspection des
# attributs internes de torch.classes. SimpleNamespace remplace le _path
# manquant sans modifier le comportement de PyTorch.
import torch, types
torch.classes.__path__ = types.SimpleNamespace(_path=[])


# ==============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================================

# Configuration de la page (titre, icône, disposition)
st.set_page_config(
    page_title="Chatbot Observatoire Astronomique",  # Titre du navigateur
    page_icon=":astronaut:",  # Icône du navigateur
    layout="wide",  # Disposition large (sidebar + contenu)
    menu_items={
        'About': "Observatoire Astronomique - IMT Atlantique, campus de Brest"
    }
)

# Charge les variables d'environnement depuis .env (override=True pour que les
# valeurs locales priment sur les variables système déjà définies).
load_dotenv(override=True)

# ==============================================================================
# GESTION DE L'ENVIRONNEMENT (Local vs Streamlit Cloud)
# ==============================================================================

# Détecte l'environnement d'exécution (local, docker ou cloud)
ENVIRONMENT = detect_environment()

# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def send_email(receiver_email, subject, body):
    """
    Envoie un email transactionnel via le serveur SMTP Gmail (port 587, TLS).

    Appelée par les actions "Mot de passe oublié" et "Nom d'utilisateur oublié"
    de la sidebar pour transmettre les informations de récupération à l'utilisateur.

    Les credentials SMTP (SMTP_USER, SMTP_PASSWORD) doivent être définis dans
    les variables d'environnement ou dans st.secrets (selon l'environnement).
    SMTP_PASSWORD doit être un "mot de passe d'application" Gmail, pas le mot
    de passe principal du compte.

    Args:
        receiver_email (str): Adresse email du destinataire.
        subject (str): Sujet de l'email.
        body (str): Corps du message en texte brut.

    Raises:
        ValueError: Si SMTP_USER ou SMTP_PASSWORD ne sont pas configurés.
        smtplib.SMTPException: En cas d'erreur de connexion ou d'envoi SMTP.
    """
    # Configuration du serveur SMTP Gmail
    smtp_host = "smtp.gmail.com"
    smtp_port = 587  # Port SMTP TLS
    # Récupérer les credentials SMTP de manière sécurisée
    smtp_user = get_env_variable("SMTP_USER")  # Email expéditeur
    smtp_pass = get_env_variable("SMTP_PASSWORD")  # Mot de passe applicatif Gmail
    
    if not smtp_user or not smtp_pass:
        raise ValueError("SMTP_USER et SMTP_PASSWORD ne sont pas configurés")
    
    # Connexion au serveur SMTP
    server = smtplib.SMTP(smtp_host, smtp_port)
    server.connect(smtp_host, smtp_port)
    server.starttls()  # Activer la sécurité TLS
    server.login(smtp_user, smtp_pass)  # Authentification

    # Création du message email
    msg = MIMEText(body)  # Corps du message en texte
    msg["Subject"] = subject  # Sujet
    msg["From"] = smtp_user  # Expéditeur
    msg["To"] = receiver_email  # Destinataire

    # Envoi du message et fermeture de la connexion
    server.send_message(msg)
    server.quit()


# ==============================================================================
# AUTHENTIFICATION DES UTILISATEURS
# ==============================================================================

# La clé API Mistral sera chargée dynamiquement après l'authentification
# pour éviter les problèmes avec st.secrets en local

# Chemin absolu vers config.yaml (deux niveaux au-dessus du dossier src/).
# Ce fichier contient les credentials chiffrés, les rôles et les emails.
config_path = Path(__file__).resolve().parent.parent / "config.yaml"

# Initialise l'authenticateur avec le fichier YAML. stauth gère le hachage
# des mots de passe, les cookies de session et les opérations CRUD sur les comptes.
authenticator = stauth.Authenticate(str(config_path))
# ------------- Register new user panel ------------------------
# Panneau de gestion des utilisateurs dans la barre latérale
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
                    # Le nouveau mot de passe temporaire a été envoyé par email.
                    # L'utilisateur devra le changer dès sa prochaine connexion.
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
                    # L'identifiant a été envoyé à l'adresse email associée au compte.
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
# ==============================================================================
# LOGIQUE PRINCIPALE DE L'APPLICATION
# ==============================================================================
# Cette section s'exécute uniquement si l'utilisateur est authentifié
elif st.session_state.get('authentication_status'):
    # Charger la clé API Mistral de manière sécurisée selon l'environnement
    api_key = get_env_variable("MISTRAL_API_KEY")
    
    if not api_key:
        st.error(f"❌ MISTRAL_API_KEY n'est pas configuré.\n\n"
                f"**Environnement détecté**: {ENVIRONMENT.upper()}\n\n"
                f"**Instructions**:\n"
                f"- En local: Ajouter `MISTRAL_API_KEY=votre_clé` dans le fichier `.env`\n"
                f"- Sur Streamlit Cloud: Ajouter le secret dans les paramètres de l'application\n"
                f"- Dans Docker: Passer la variable via `-e MISTRAL_API_KEY=votre_clé`")
        st.stop()

    # ==============================================================================
    # RÉPERTOIRES DE TRAVAIL
    # ==============================================================================
    # Définit les chemins pour les documents, index vectoriel et historiques
    DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"  # Documents PDF
    INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"  # Index FAISS
    HISTORY_DIR = Path(__file__).resolve().parent.parent / "chat_histories"  # Historiques

    # Crée les répertoires s'ils n'existent pas
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    # ==============================================================================
    # INFORMATIONS DE L'UTILISATEUR CONNECTÉ
    # ==============================================================================
    # Récupère le nom et les rôles de l'utilisateur depuis la session Streamlit
    user_name = st.session_state.get("name")
    user_roles = st.session_state.get("roles")
  

    # Convertit le nom d'affichage en identifiant de dossier filesystem-safe :
    # minuscules + espaces → underscores (ex. "Jean Dupont" → "jean_dupont").
    file_safe_name = user_name.lower().replace(' ', '_')
    user_history_path = HISTORY_DIR / file_safe_name
    os.makedirs(user_history_path, exist_ok=True)

    # ==============================================================================
    # INITIALISATION DU MODÈLE ET DE LA BASE VECTORIELLE
    # ==============================================================================
    # Initialise le modèle de langage et la base vectorielle FAISS
    model, embedding_function = model_and_embedding_function(api_key)
    contextualize_q_prompt = create_contextualize_q_system_prompt()
    
    # Initialise le mode raisonnement s'il n'existe pas en session
    if "reasoning_mode" not in st.session_state:
        st.session_state.reasoning_mode = False
    
    # Crée le prompt en fonction du mode raisonnement (mis à jour dynamiquement)
    prompt = create_prompt(reasoning_mode=st.session_state.get("reasoning_mode", False))

    # Charge la base vectorielle FAISS (ou None si elle n'existe pas)
    if "vector" not in st.session_state:
        st.session_state.vector = load_vector_store(INDEX_DIR, embedding_function)
    
    # Détermine si la chaîne RAG doit être reconstruite.
    # Deux conditions déclenchent une reconstruction :
    #  1. La chaîne n'existe pas encore en session (premier chargement).
    #  2. Le mode raisonnement a changé depuis la dernière construction
    #     (le prompt système diffère selon ce mode, ce qui rend la chaîne obsolète).
    rebuild_chain = False
    if "chain" not in st.session_state:
        rebuild_chain = True
    elif "last_reasoning_mode" in st.session_state:
        if st.session_state.last_reasoning_mode != st.session_state.reasoning_mode:
            rebuild_chain = True
    
    # Reconstruit la chaîne si nécessaire
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
            # Mémorise le mode de raisonnement pour lequel cette chaîne a été construite
            st.session_state.last_reasoning_mode = st.session_state.reasoning_mode

    # ==============================================================================
    # INITIALISATION DE L'ÉTAT DE SESSION
    # ==============================================================================
    # Streamlit reexécute le script complet à chaque interaction utilisateur.
    # st.session_state permet de conserver des valeurs entre ces reruns.
    # Le pattern "if X not in st.session_state" évite de réinitialiser des
    # variables déjà définies lors des reruns suivants.
    if "available_documents" not in st.session_state:
        st.session_state.available_documents = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "history_path" not in st.session_state:
        # Crée un nouveau fichier d'historique avec timestamp
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state.history_path = user_history_path / f"{ts}.json"

    # ==============================================================================
    # FONCTIONS AUXILIAIRES
    # ==============================================================================
    
    def refresh_document_list():
        """
        Synchronise st.session_state.available_documents avec le contenu de DOCS_DIR.

        Parcourt le répertoire docs/ et collecte tous les fichiers .pdf/.PDF.
        Le résultat est stocké dans st.session_state.available_documents, qui
        alimente les selectbox de suppression et l'expander "Documents disponibles".

        Cette fonction est appelée après chaque upload ou suppression de document
        pour maintenir la liste à jour sans recharger toute la page.
        """
        if DOCS_DIR.exists():
            # Récupère tous les fichiers PDF du répertoire docs
            st.session_state.available_documents = [f for f in os.listdir(DOCS_DIR) if f.endswith(('.pdf', '.PDF'))]
        else:
            st.session_state.available_documents = []

    def save_chat_history():
        """
        Persiste l'historique de la conversation courante dans un fichier JSON.

        Sauvegarde deux structures complémentaires :
        - "messages"     : liste des dicts {role, content, sources} affichés dans l'UI.
        - "chat_history" : liste de dicts {type, content} reconstituant les objets
                           HumanMessage/AIMessage passés à la chaîne RAG.

        Le fichier est nommé par le timestamp de création de la session
        (ex. "2025-06-01_20-00-00.json") et placé dans
        chat_histories/{file_safe_name}/ (nom de l'utilisateur en snake_case).

        Appelée automatiquement après chaque échange utilisateur/assistant.

        Returns:
            bool: True si l'écriture a réussi, False si une exception s'est produite
                  (l'erreur est également affichée via st.error).
        """
        try:
            # Prépare les données à sauvegarder
            history_data = {
                "messages": st.session_state.messages,  # Messages affichés à l'interface
                "chat_history": [  # Historique pour le RAG
                    {"type": "human", "content": msg.content} if isinstance(msg, HumanMessage)
                    else {"type": "ai", "content": msg.content}
                    for msg in st.session_state.chat_history
                ]
            }
            
            # Convertit le Path en string pour la sérialisation JSON
            history_path_str = str(st.session_state.history_path)
            
            # Crée les répertoires parents s'ils n'existent pas
            os.makedirs(os.path.dirname(history_path_str), exist_ok=True)
            
            # Sauvegarde avec encodage UTF-8
            with open(history_path_str, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde: {e}")
            return False

    # Initialize document list
    refresh_document_list()

    # ==============================================================================
    # INTERFACE DE PRÉSENTATION ET DE BIENVENUE
    # ==============================================================================

    # Affiche le logo centré dans la colonne du milieu (ratio 1:2:1).
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        logo_path = Path(__file__).resolve().parent.parent / "assets" / "images" / "logo-gens-de-la-lune.png"
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        else:
            st.info("Logo 'Gens de la Lune' non trouvé. Veuillez placer l'image dans assets/images/")

    # Add a title
    st.title("Je suis **Astrobot** 🤖, ton assistant virtuel pour l'observatoire 🔭")

    st.markdown(f"""
    Salut **{user_name}** ! 👋  
    Je peux t'aider avec l'utilisation et la configuration des équipements.
    Pose-moi une question ou demande-moi de l'aide !
    """)

    with st.sidebar:
        # ==============================================================================
        # OPTIONS DE PARAMÉTRAGE
        # ==============================================================================
        # Mode raisonnement toggle
        st.subheader("⚙️ Options")
        if "reasoning_mode" not in st.session_state:
            st.session_state.reasoning_mode = False
        
        # Bascule pour activer/désactiver le mode raisonnement détaillé
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
        
        # ==============================================================================
        # GESTION DE L'HISTORIQUE DE DISCUSSION
        # ==============================================================================
        with st.expander("Historique de la discussion", expanded=False):
            if "admin" in user_roles:
                # Les admins voient les historiques de tous les utilisateurs.
                # On liste les sous-dossiers de HISTORY_DIR, chacun correspondant
                # à un utilisateur (nom en snake_case).
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
                        history_file_path = user_history_path / f"{selected_history}.json"
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

        # ==============================================================================
        # GESTION DES DOCUMENTS
        # ==============================================================================
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
                
                # Sélectionne l'embedder approprié selon le choix de l'utilisateur
                embedder = Embedder(api_key)  # Embedder par défaut
                if selected_embedder == "Text only embedder":
                    embedder = TextEmbedder(api_key)
                elif selected_embedder == "Embedder with OCR":
                    embedder = EmbedderWithOcr(api_key)
                elif selected_embedder == "Multimodal embedder":
                    embedder = MultimodalEmbedder(api_key)
                add_permanently = False
                if "admin" in user_roles:
                    # Seul un admin peut sauvegarder l'index FAISS sur disque.
                    # Sans cette option, le document est indexé en mémoire pour la
                    # session courante uniquement (perdu au prochain redémarrage).
                    add_permanently = st.checkbox("Ajouter le document de manière permanente")
                if st.button("Traiter", key="process_button"):
                    try:
                        # Construit le chemin de destination et sauvegarde le fichier
                        save_path = os.path.join(DOCS_DIR, pdf.name)
                        with open(save_path, "wb") as f:
                            f.write(pdf.read())
                        if "admin" in user_roles:
                            st.success(f"Fichier enregistré dans : {save_path}")
                        with st.spinner("Indexation du document…"):
                            # Traite le document et met à jour l'index vectoriel
                            st.session_state.vector = embedder.embed(save_path, st.session_state.vector, save=add_permanently)
                            # Reconstruit la chaîne avec le nouvel index
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
                            # Supprime les vecteurs FAISS associés au document.
                            # Note : "delection_state" est une faute de frappe héritée
                            # du code original ; conserver tel quel pour ne pas casser
                            # d'éventuelles références externes.
                            delection_state = Embedder(api_key).delete_document(selected_doc, st.session_state.vector, save=delete_permanently)
                            if delection_state is False:
                                st.error(f"Le document '{selected_doc}' n'existe pas ou n'a pas pu être supprimé.")
                            else:
                                if delete_permanently:
                                    os.remove(file_path)
                                # Reconstruit la chaîne avec l'index mis à jour
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


    # ==============================================================================
    # AFFICHAGE DE L'HISTORIQUE DE CHAT
    # ==============================================================================
    st.header("Conversation")

    # Affiche les messages précédents de la conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Affiche les documents utilisés pour cette réponse
            if message.get("role") == "assistant" and message.get("sources"):
                with st.expander("📚 Documents utilisés"):
                    # Dictionnaire pour éviter les doublons
                    unique_sources = {}
                    for item in message["sources"]:
                        source = item.get("source") or "Unknown"
                        content = item.get("content") or ""
                        if source not in unique_sources:
                            unique_sources[source] = content

                    for i, (source, content) in enumerate(unique_sources.items(), 1):
                        # Extrait juste le nom de fichier du chemin
                        if source != 'Unknown':
                            filename = source.split('\\')[-1].split('/')[-1]
                        else:
                            filename = source
                        st.write(f"**{i}. {filename}**")
                        st.write(f"```\n{content}\n```")

    # ==============================================================================
    # QUESTIONS SUGGÉRÉES
    # ==============================================================================
    st.markdown("**💡 Questions suggérées :**")
    col1, col2, col3, col4 = st.columns(4)
    
    # Liste des questions suggestions pré-définies
    suggested_questions = [
        "Quel est le programme ce soir?",
        "Quelle est la météo actuelle?",
        "Comment utiliser le télescope?",
        "Quels sont les objets de Messier visibles ce soir?"
    ]
    
    # question_clicked reçoit le texte de la question si l'un des boutons
    # est pressé. Il est ensuite passé à st.chat_input comme valeur par défaut,
    # ce qui unifie le traitement des questions suggérées et de la saisie libre.
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

    # ==============================================================================
    # INTERFACE DE CHAT
    # ==============================================================================
    # Entrée texte pour les questions de l'utilisateur
    user_input = st.chat_input("Posez votre question ici...")

    # Si un bouton de question suggérée est cliqué, utiliser cette question
    if question_clicked:
        user_input = question_clicked

    # Traite la question si l'utilisateur a saisi du texte
    if user_input:
        # 1) Affiche et enregistre le message utilisateur
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 2) Crée un conteneur vide qui sera réutilisé pour afficher les tentatives
        #    en cours puis la réponse finale. st.empty() garantit que chaque réécriture
        #    remplace le contenu précédent au lieu d'empiler de nouveaux éléments.
        assistant_slot = st.empty()

        # 3) Boucle de génération avec retry automatique.
        #    En cas d'erreur API (timeout, rate limit…), la boucle réessaie
        #    indéfiniment en affichant le numéro de tentative dans le placeholder.
        while True:
            nb_tentatives = 0
            try:
                with st.spinner("Réflexion en cours..."):
                    # Utilise le mode raisonnement si activé
                    reasoning_mode = st.session_state.get("reasoning_mode", False)
                    response, documents, messier_images = get_response(
                        user_input, 
                        st.session_state["chat_history"], 
                        st.session_state.vector, 
                        st.session_state.chain,
                        reasoning_mode=reasoning_mode  # Passe le mode raisonnement
                    )
                break  # Sort de la boucle si tout fonctionne
            except Exception as e:
                nb_tentatives += 1
                # Réécrit dans le même conteneur → l'ancien texte est remplacé
                with assistant_slot.chat_message("assistant"):
                    st.write(f"Une erreur est survenue : {e}. Nouvelle tentative…(Tentative numéro : {nb_tentatives})")

        # 4) Réponse finale : remplace le placeholder par le contenu réel
        with assistant_slot.chat_message("assistant"):
            # rendered_messier_blocks : flag qui indique si les blocs Messier ont déjà
            # été rendus côte-à-côte avec leurs images. S'il vaut True, le st.write(response)
            # générique est ignoré pour éviter d'afficher la réponse en double.
            rendered_messier_blocks = False
            if messier_images:
                # Découpe la réponse en blocs séparés par des lignes vides.
                # Chaque bloc correspond à un objet Messier et sera affiché
                # à gauche, avec l'image associée à droite.
                blocks = [b.strip() for b in response.split("\n\n") if b.strip()]
                if blocks:
                    for idx, block in enumerate(blocks, 1):
                        col_info, col_img = st.columns([0.7, 0.3], gap="medium")
                        with col_info:
                            st.markdown(block.replace("\n", "  \n"))
                        with col_img:
                            if idx <= len(messier_images):
                                img_data = messier_images[idx - 1]
                                st.image(img_data['image_path'], caption=img_data['source'])
                        if idx < len(blocks):
                            st.markdown("---")
                    rendered_messier_blocks = True

            if not rendered_messier_blocks:
                st.write(response)

            # Display Messier page top 10 if available
            messier_page_data = None
            if documents:
                for doc in documents:
                    if hasattr(doc, 'metadata'):
                        if doc.metadata.get('source') == 'messier.astronomie-pointedudiable.fr' and doc.metadata.get('type') == 'messier_page_top10':
                            messier_page_data = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                            break

            if messier_page_data and "LISTE DES 10 OBJETS" in messier_page_data:
                st.markdown("---")
                st.subheader("🗂️ Objets Messier (page publique)")
                rows = []
                for line in messier_page_data.splitlines():
                    if line.strip().startswith(tuple(str(i) + "." for i in range(1, 11))):
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) >= 6:
                            idx_and_m = parts[0].split(".", 1)
                            messier_label = idx_and_m[1].strip() if len(idx_and_m) > 1 else parts[0].strip()
                            rows.append({
                                "Messier": messier_label,
                                "Objet": parts[1],
                                "Saison": parts[2].replace("Saison:", "").strip(),
                                "Magnitude": parts[3].replace("Mag:", "").strip(),
                                "Constellation": parts[4].replace("Constellation:", "").strip(),
                                "Visible": parts[5].replace("Visible:", "").strip()
                            })

                if rows:
                    st.table(rows)
                else:
                    st.info("Aucun objet Messier n'a été extrait de la page publique.")
            
            # Display Messier images if available (fallback if not rendered alongside blocks)
            if messier_images and not rendered_messier_blocks:
                st.markdown("---")
                st.subheader("🔭 Photographies des objets Messier")
                
                for idx, img_data in enumerate(messier_images, 1):
                    try:
                        col_info, col_img = st.columns([0.7, 0.3], gap="medium")
                        
                        with col_info:
                            st.markdown(f"**{idx}. {img_data['messier_label']}**")
                            info_text = img_data.get("info")
                            if info_text:
                                st.markdown(info_text)
                            else:
                                st.markdown("*Aucune information trouvée dans Catalogue Messier.pdf pour cet objet.*")
                        
                        with col_img:
                            st.image(img_data['image_path'], caption=img_data['source'])
                        
                        st.markdown("---")
                    except Exception as e:
                        st.warning(f"Erreur lors de l'affichage de l'image {img_data['messier_label']}: {e}")
            
            # Construit la liste des sources pour l'expander "Documents utilisés".
            # sources_payload est aussi stocké dans st.session_state.messages pour
            # être réaffiché lors des reruns (historique de conversation).
            sources_payload = []
            if documents:
                with st.expander("📚 Documents utilisés"):
                    # Déduplique par nom de source : si plusieurs chunks proviennent
                    # du même fichier, seul le premier est conservé pour l'affichage.
                    unique_sources = {}
                    for doc in documents:
                        # Essaie différentes façons de récupérer la source
                        source = None
                        if hasattr(doc, 'metadata'):
                            source = doc.metadata.get('source')
                            if not source:
                                # Essaie d'autres clés possibles
                                source = doc.metadata.get('file') or doc.metadata.get('filename') or doc.metadata.get('path')
                        
                        if not source:
                            source = 'Unknown'
                        
                        # Extrait le contenu du document
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        sources_payload.append({"source": source, "content": content})
                        
                        if source not in unique_sources:
                            unique_sources[source] = content
                    
                    for i, (source, content) in enumerate(unique_sources.items(), 1):
                        # Extrait juste le nom de fichier du chemin
                        if source != 'Unknown':
                            filename = source.split('\\')[-1].split('/')[-1]
                        else:
                            filename = source
                        st.write(f"**{i}. {filename}**")
                        st.write(f"```\n{content}\n```")

        # 5) Historique pour le RAG et pour la page
        # Ajoute le message et la réponse à l'historique interne
        st.session_state["chat_history"].extend(
            [HumanMessage(content=user_input), AIMessage(content=response)]
        )
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources_payload})

        # Sauvegarde l'historique après chaque échange
        save_chat_history()
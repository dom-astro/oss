# ============================================================================
# IMPORTS
# ============================================================================
# Document: Schéma pour la création d'objets Document contenant du contenu textuel
from langchain.schema import Document
# RecursiveCharacterTextSplitter: Outil de division de texte par caractères récursifs
from langchain.text_splitter import RecursiveCharacterTextSplitter
# ChatMistralAI: Interface pour accéder aux modèles de chat Mistral (ex: mistral-large-latest)
from langchain_mistralai.chat_models import ChatMistralAI
# ChatPromptTemplate: Création de templates de prompts réutilisables
from langchain_core.prompts import ChatPromptTemplate
# StrOutputParser: Parseur pour convertir les sorties du modèle en chaînes de caractères
from langchain_core.output_parsers import StrOutputParser
# Mistral: Client Mistral pour les requêtes API directes (notamment vision pixtral-large-latest)
from mistralai import Mistral
# partition_pdf: Fonction d'Unstructured pour extraire les éléments structurés d'un PDF
from unstructured.partition.pdf import partition_pdf
# Embedder: Classe parent fournissant les fonctionnalités de base d'embeddings FAISS
from Embedder import Embedder
# base64: Encodage/décodage Base64 pour les images
import base64
# io: Entrées/sorties en mémoire pour les opérations sur les fichiers
import io
# Image: Module PIL pour manipuler les images
from PIL import Image
# Utilitaires de retry: Mécanismes de tentatives avec attente exponentielle
from tenacity import retry, stop_after_attempt, wait_incrementing, retry_if_exception_type


# ============================================================================
# CLASSE MULTIMODALEMBEDDER
# ============================================================================
class MultimodalEmbedder(Embedder):
    """
    Embedder multimodal avancé pour traiter des PDFs complexes.
    
    RESPONSABILITÉS:
    ----------------
    - Extraction et traitement de documents PDF contenant du texte, des images ET des tableaux
    - Génération de résumés intelligents pour chaque type d'élément via l'IA
    - Utilisation du modèle de vision pixtral-large-latest pour analyser les images
    - Synthèse contextuelle des images avec le texte environnant
    - Conversion des tableaux en résumés textuels enrichis via LLM
    
    CAPACITÉS MULTIMODALES:
    ----------------------
    1. Texte natif: Extraction directe du contenu textuel
    2. Images: Analyse via le modèle de vision pixtral-large, avec contexte avant/après
    3. Tableaux: Extraction HTML et résumé via ChatMistralAI (mistral-large-latest)
    
    UTILISATION:
    -----------
    - Documents complexes: Rapports techniques, articles scientifiques, documents mixtes
    - Archivage intelligent: Vectorisation optimisée pour la récupération sémantique (RAG)
    - Indexation multimédia: Recherche pertinente sur contenu textuel, visuel ET tabulaire
    
    ATTRIBUTS HÉRITÉS:
    -----------------
    - api_key: Clé API Mistral pour l'accès aux modèles
    - faiss_path: Chemin vers l'index FAISS
    - document_index: Dictionnaire UUID -> métadonnées document
    
    NOTE: Cette classe hérite de Embedder et étend ses fonctionnalités
          avec le traitement multimodal intelligent.
    """
    
    def __init__(self, api_key):
        """
        Initialise l'embedder multimodal.
        
        PROCESSUS:
        ----------
        1. Appelle le constructeur parent (Embedder) pour initialiser:
           - La clé API Mistral
           - Le chemin FAISS
           - L'index de documents
        2. Hérite de toutes les méthodes d'embedding FAISS du parent
        
        ARGS:
        ----
        api_key (str): Clé API Mistral pour accéder aux modèles (chat, vision, embedding)
        
        EXEMPLE:
        -------
        >>> embedder = MultimodalEmbedder(api_key="your-api-key")
        """
        # Appel du constructeur parent pour initialiser la clé API et les chemins FAISS
        super().__init__(api_key)
    
    # ========================================================================
    # MÉTHODE DE RÉSUMÉ DE TABLEAUX
    # ========================================================================
    def summarize_Table(self, item):
        """
        Résume un tableau ou un bloc de texte en utilisant un modèle de chat LLM.
        
        PROCESSUS:
        ----------
        1. Crée une instance du modèle ChatMistralAI (mistral-large-latest)
        2. Définit un prompt d'instruction pour la résumé optimisé RAG
        3. Construit une chaîne LangChain: prompt → modèle → parseur de sortie
        4. Invoque la chaîne avec le contenu du tableau/texte
        5. Gère les erreurs réseau avec une boucle de retry simple
        
        ARGS:
        ----
        item (str): Contenu du tableau (HTML) ou texte à résumer
        
        RETURNS:
        -------
        str: Résumé concis du tableau en français, optimisé pour l'embedding
        
        NOTES:
        -----
        - Le prompt force la sortie en français
        - Les résumés sont optimisés pour la récupération sémantique (RAG)
        - La boucle while gère les timeouts et erreurs API Mistral
        - Il y a du code de retry en commentaire (approche par décorateur)
        
        EXEMPLE:
        -------
        >>> embedder = MultimodalEmbedder(api_key="...")
        >>> html_table = "<table><tr><td>Données</td></tr></table>"
        >>> summary = embedder.summarize_Table(html_table)
        """
        # Crée une nouvelle instance du modèle ChatMistralAI (mistral-large-latest)
        text_model = ChatMistralAI(mistral_api_key=self.api_key, model="mistral-large-latest")
        
        # Définit le prompt d'instruction pour la résumé optimisé
        # - Force la sortie en français
        # - Optimise pour l'embedding RAG (pas d'introduction, juste le contenu)
        # - Demande un résumé concis et pertinent
        prompt_text = """
        You are an assistant tasked with summarizing tables and text.
        Give a concise summary of the table or text.
        The summary must be in french.
        Respond only with the summary, no additionnal comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.
        Optimize your summary embedding as it will be used for RAG.
        Table or text chunk: {element}
        """
        
        # Crée un template de prompt réutilisable
        prompt = ChatPromptTemplate.from_template(prompt_text)
        
        # Crée une chaîne LangChain: lambda (identité) → prompt → modèle → parseur
        # Cette composition permet: entrée → formatage → génération → extraction texte
        summarize_chain = {"element": lambda x: x} | prompt | text_model | StrOutputParser()
        
        # BOUCLE DE RETRY: Gère les erreurs réseau et timeouts API
        # Approche simple avec retry infini jusqu'au succès
        # Code en commentaire montre une approche par décorateur (plus robuste mais actuellement désactivée)
        #@retry(
        #    retry=retry_if_exception_type(Exception),
        #    wait=wait_incrementing(start=30, increment=30, max=120),
        #    stop=stop_after_attempt(5)
        #)
        #def _invoke_chain(chain, item):
            #return chain.invoke({"element": item})
        #return _invoke_chain(summarize_chain, item)
        
        # Boucle infinie jusqu'au succès (retry simple)
        while True:
            try:
                # Appel de la chaîne LangChain pour invoquer le modèle Mistral
                response = summarize_chain.invoke({"element": item})
                break  # Sort de la boucle si succès
            except Exception as e:
                # Affiche l'erreur et réessaye (reconnexion automatique)
                print(f"Error: {e}")
        
        # Retourne le résumé généré par le modèle
        return response

    # ========================================================================
    # MÉTHODE DE RÉSUMÉ D'IMAGES
    # ========================================================================
    def summarize_image(self, b64, prefix, suffix):
        """
        Résume une image extraite d'un PDF en utilisant le modèle de vision pixtral-large.
        
        PROCESSUS:
        ----------
        1. Crée un client Mistral pour accéder au modèle de vision
        2. Construit un prompt détaillé qui:
           - Donne le contexte textuel avant et après l'image
           - Force la sortie en français et concise
           - Optimise pour la récupération sémantique (RAG)
           - Demande une description basée sur ce qui est visible
        3. Encode l'image en Base64 pour la transmission API
        4. Invoque le modèle pixtral-large-latest avec image + texte
        5. Gère les erreurs avec une boucle de retry simple
        
        ARGS:
        ----
        b64 (str): Image encodée en Base64 (extraite du PDF)
        prefix (str): Texte français immédiatement avant l'image (contexte)
        suffix (str): Texte français immédiatement après l'image (contexte)
        
        RETURNS:
        -------
        str: Paragraph en français (≤150 mots) décrivant uniquement le contenu de l'image
        
        NOTES:
        -----
        - Le modèle de vision utilisé est pixtral-large-latest (meilleure qualité)
        - Les images du logo "Gens de la lune" sont ignorées (exclusion)
        - Le prompt force une description purement visuelle, sans répétition du texte
        - La sortie est optimisée pour l'embedding (concise, pertinente, orientée RAG)
        - Les images < 2x2 pixels sont filtrées dans load_and_split()
        
        EXEMPLE:
        -------
        >>> embedder = MultimodalEmbedder(api_key="...")
        >>> b64_img = "iVBORw0KG..." # Base64 image
        >>> description = embedder.summarize_image(b64_img, "Avant", "Après")
        """
        # Crée un client Mistral pour accéder aux modèles d'API (notamment pixtral-large)
        client = Mistral(api_key=self.api_key)
        
        # Construit le prompt détaillé pour la vision avec instructions précises
        # Ce prompt force:
        # - Une description purement visuelle (ignorer le texte environnant)
        # - La suppression des images de logo "Gens de la lune"
        # - Une sortie concise, optimisée pour RAG
        # - Langage français uniquement
        prompt = f""" You are given:
        - an inline image (base-64) extracted from a PDF
        - the French text immediately before and after the image

        Text-before:
        {prefix}

        Text-after:
        {suffix}

        Task:
        - Write **one paragraph (≤ 150 words, in French)** that describes only what is visible in the image.
        - If the image is the logo of "Association Gens de la lune", do not describe it.
        - Use the surrounding text only to resolve names, labels or context; do not repeat or paraphrase it.
        - Keep the prose concise and optimised for semantic retrieval (RAG).
        - Your priority is to describe the image, not to summarize the text.
        - Do not mention that you are describing an image, and do not start with phrases like “Cette image montre”.
        - Do not include any additional information or context that is not visible in the image.
        - If the image is not relevant to the text, do not describe it.
        Respond in French.  Output only the paragraph."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ]
        
        # Code de retry en commentaire (approche par décorateur, désactivée)
        #@retry(
        #    retry=retry_if_exception_type(Exception),
        #    wait=wait_incrementing(start=30, increment=30, max=120),
        #    stop=stop_after_attempt(5)
        #)
        #def _chat_with_retry():
        
        # Boucle infinie jusqu'au succès (retry simple)
        while True:
            try:
                # Appel du modèle de vision pixtral-large-latest avec image + contexte
                response = client.chat.complete(
                    model="pixtral-large-latest",
                    messages=messages,
                )
                break  # Sort de la boucle si succès
            except Exception as e:
                # Affiche l'erreur et réessaye (reconnexion automatique)
                print(f"Error: {e}")

        # Code de retry en commentaire
        #response = _chat_with_retry()
        
        # Retourne le contenu du message généré par le modèle de vision
        return response.choices[0].message.content

    # ========================================================================
    # MÉTHODE DE CONSTRUCTION DE CONTEXTE POUR IMAGES
    # ========================================================================
    
    def image_wrapper(self, core_elems, text_list, i, max_chars=200, elements_nb=3):
        """
        Construit le contexte textuel (avant/après) pour une image.
        
        PROCESSUS:
        ----------
        1. Construit le PRÉFIXE à partir des derniers N éléments textuels
           - Extrait du texte_list (éléments déjà traités)
           - Tronque à max_chars caractères
        2. Construit le SUFFIXE à partir des N prochains éléments
           - Ignore les images et tableaux (seul le texte)
           - Tronque à max_chars caractères
        3. Retourne le couple (préfixe, suffixe) pour le résumé image
        
        ARGS:
        ----
        core_elems (list): Liste de tous les éléments du document PDF
        text_list (list): Liste des textes déjà traités (historique)
        i (int): Index courant dans core_elems (position de l'image)
        max_chars (int): Nombre max de caractères pour prefix/suffix (défaut: 200)
        elements_nb (int): Nombre d'éléments à considérer pour le contexte (défaut: 3)
        
        RETURNS:
        -------
        tuple: (prefix, suffix) - Pair de contextes textuels avant/après
        
        NOTES:
        -----
        - Le préfixe est construit RÉTROSPECTIVEMENT à partir de text_list
        - Le suffixe est construit PROSPECTIVEMENT à partir de core_elems
        - Les images et tableaux du suffixe sont filtrés
        - max_chars tronque les contextes pour ne pas surcharger le prompt vision
        
        EXEMPLE:
        -------
        >>> core_elems = [Document1, Image, Document2, ...]
        >>> text_list = ["Doc1 texte", "Doc2 texte"]
        >>> prefix, suffix = embedder.image_wrapper(core_elems, text_list, 1)
        """
        # Construit le préfixe à partir des derniers N éléments du text_list
        # text_list contient les textes déjà traités (historique du document)
        prefix = " \n".join(text_list[-elements_nb:])
        # Tronque le préfixe à max_chars caractères pour ne pas surcharger le prompt vision
        prefix = prefix[-max_chars:]

        # Construit le suffixe à partir des N prochains éléments du core_elems
        # Filtre pour ne garder que le texte (pas les images ni les tableaux)
        suffix = [
            el.text for el in core_elems[i:i + elements_nb]
            if el.category not in {"Image", "Table"}
        ]
        # Joint les éléments textuels avec des sauts de ligne
        suffix = " \n".join(suffix)
        # Tronque le suffixe à max_chars caractères
        if len(suffix) > max_chars:
            suffix = suffix[:max_chars]

        # Retourne le couple (préfixe, suffixe) pour le contexte de l'image
        return prefix, suffix

    # ========================================================================
    # MÉTHODE D'EXTRACTION D'ÉLÉMENTS STRUCTURÉS DU PDF
    # ========================================================================
    def get_core_elements(self, file_path):
        """
        Extrait les éléments structurés d'un PDF avec détection de tableaux et images.
        
        PROCESSUS:
        ----------
        1. Utilise Unstructured.partition_pdf avec stratégie hi_res
        2. Active l'inférence de structure de tableaux (HTML)
        3. Extrait les images en Base64 pour transmission API
        4. Filtre les en-têtes, pieds de page, numéros de pages
        5. Retourne liste d'éléments structurés et catégorisés
        
        ARGS:
        ----
        file_path (str): Chemin absolu du fichier PDF à traiter
        
        RETURNS:
        -------
        list: Liste d'éléments structurés (Document, Image, Table) sans headers/footers
        
        NOTES:
        -----
        - "hi_res" strategy: Meilleure détection de structure (nécessaire pour tableaux)
        - infer_table_structure=True: Convertit les tableaux en HTML
        - extract_image_block_to_payload=True: Encode les images en Base64 (pas sur disque)
        - Les images/tableaux sont sauvegardés en Base64, pas en fichiers temporaires
        - Les headers, footers, page numbers sont supprimés pour le nettoyage
        
        EXEMPLE:
        -------
        >>> embedder = MultimodalEmbedder(api_key="...")
        >>> elements = embedder.get_core_elements("/chemin/to/document.pdf")
        >>> for elem in elements:
        >>>     print(elem.category)  # "Text", "Image", "Table"
        """
        # Appel Unstructured pour extraire tous les éléments du PDF
        # Stratégie hi_res: meilleure détection de structure, plus lent mais plus précis
        elements = partition_pdf(
            filename=file_path,
            infer_table_structure=True,            # Extrait les tableaux en HTML
            strategy="hi_res",                     # Stratégie haute résolution (nécessaire tableaux)
            extract_image_block_types=["Image"],   # Extrait les images (pas les images de tableaux)
            # image_output_dir_path=output_path,   # Si None: sauvegarde en Base64 (pas sur disque)
            extract_image_block_to_payload=True,   # Vrai: encode images en Base64 pour API
        )

        # Filtre les éléments non pertinents pour la vectorisation
        # Supprime: en-têtes, pieds de page, numéros de pages (bruit)
        # Filtre les éléments non pertinents pour la vectorisation
        # Supprime: en-têtes, pieds de page, numéros de pages (bruit)
        core_elems = [
            el for el in elements
            if el.category not in ("Header", "Footer", "PageNumber")
        ]
        # Retourne la liste des éléments filtrés (prêt pour traitement)
        return core_elems
    
    # ========================================================================
    # MÉTHODE PRINCIPALE: CHARGEMENT, TRAITEMENT ET SPLITTING
    # ========================================================================
    def load_and_split(self, file_path):
        """
        Traite les éléments d'un PDF, résume images et tableaux, et découpe en chunks.
        
        PROCESSUS:
        ----------
        1. Extrait les éléments structurés du PDF via get_core_elements()
        2. Initialise un splitter de texte (chunk_size=1000, overlap=200)
        3. Itère sur chaque élément et:
           a. Image: Extrait Base64 → Génère contexte (prefix/suffix) → Résume via vision
           b. Table: Extrait HTML → Résume via ChatMistralAI (résumé LLM)
           c. Texte: Ajoute directement au contenu combiné
        4. Wrap le texte combiné dans un Document LangChain
        5. Découpe en chunks cohérents avec récurrence (taille 1000, chevauchement 200)
        6. Retourne liste des chunks prêts pour l'embedding
        
        ARGS:
        ----
        file_path (str): Chemin absolu du fichier PDF à traiter
        
        RETURNS:
        -------
        list[Document]: Liste de chunks LangChain prêts pour l'embedding FAISS
        
        PROCESSUS DÉTAILLÉ:
        ------------------
        Pour chaque élément du document:
        
        - Image:
          1. Décode Base64 en image PIL
          2. Filtre les images < 2x2 pixels (bruit/artefacts)
          3. Extrait contexte: text AVANT (préfixe) + text APRÈS (suffixe)
          4. Invoque modèle vision: pixtral-large-latest
          5. Ajoute résumé image au texte combiné
        
        - Table (HTML):
          1. Extrait métadonnées: HTML brut du tableau
          2. Invoque modèle LLM: ChatMistralAI (mistral-large-latest)
          3. Ajoute résumé tableau au texte combiné
          4. Ajoute au text_list pour contexte images suivantes
        
        - Texte (autre):
          1. Ajoute directement au texte combiné
          2. Sauvegarde dans text_list pour contexte images/tableaux suivants
        
        Après itération:
        - Wrap le texte combiné dans Document(page_content=text)
        - Découpe en chunks avec RecursiveCharacterTextSplitter
        - Retourne liste de chunks (Document objects)
        
        NOTES:
        -----
        - Les chunks se chevauchent de 200 caractères (meilleure récupération)
        - La taille 1000 équilibre contexte et granularité
        - text_list garde trace de l'historique pour contexte image
        - Filtrage < 2x2 pixels évite les artefacts et logos minuscules
        
        EXEMPLE:
        -------
        >>> embedder = MultimodalEmbedder(api_key="...")
        >>> chunks = embedder.load_and_split("/chemin/to/document.pdf")
        >>> for chunk in chunks:
        >>>     print(chunk.page_content[:100])  # Aperçu du chunk
        >>> # Ensuite, utiliser chunks pour embedding FAISS:
        >>> # embedder.embed(chunks, doc_id="mon_doc")
        """
        # Extrait les éléments structurés du PDF (texte, images, tableaux)
        core_elems = self.get_core_elements(file_path)
        
        # Initialise le splitter de texte avec taille=1000 et chevauchement=200
        # Cela assure une bonne granularité tout en gardant du contexte
        # Initialise le splitter de texte avec taille=1000 et chevauchement=200
        # Cela assure une bonne granularité tout en gardant du contexte
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # Initialise le texte combiné (accumulateur) et l'historique des textes
        text = ""                    # Texte complet à découper
        text_list = []               # Historique des textes pour contexte image
        
        # Itère sur chaque élément du document (texte, image, tableau, etc.)
        for i, element in enumerate(core_elems):
            if element.category == "Image":
                # BRANCHE IMAGE: Extraire Base64 → Résumer avec vision
                
                # Extrait l'image encodée en Base64 depuis les métadonnées
                b64 = element.metadata.image_base64
                # Décode le Base64 en bytes, puis en objet PIL Image
                img_data = base64.b64decode(b64)
                image = Image.open(io.BytesIO(img_data))
                
                # FILTRE: Saute les images < 2x2 pixels (logos, artefacts, bruit)
                if image.height <= 1 or image.width <= 1:
                    continue  # Passe à l'élément suivant
                
                # Extrait le contexte textuel (avant et après l'image)
                # Cela aide le modèle de vision à mieux comprendre l'image
                prefix, suffix = self.image_wrapper(core_elems, text_list, i)
                
                # Invoque le modèle de vision pixtral-large pour résumer l'image
                image_summary = self.summarize_image(b64, prefix, suffix)
                
                # Ajoute le résumé de l'image au texte combiné
                text += image_summary + "\n"
                
            elif element.category == "Table":
                # BRANCHE TABLEAU: Extraire HTML → Résumer avec LLM
                
                # Extrait le HTML brut du tableau depuis les métadonnées
                html = element.metadata.text_as_html
                
                # Invoque le modèle LLM ChatMistralAI pour résumer le tableau
                table_summary = self.summarize_Table(html)
                
                # Ajoute le résumé du tableau au texte combiné
                text += table_summary + "\n"
                
                # Ajoute le résumé au text_list pour contexte des images/tableaux suivants
                text_list.append(table_summary)
                
            else:
                # BRANCHE TEXTE: Ajouter directement au contenu
                # Ajoute le texte brut au texte combiné (pas de traitement supplémentaire)
                text += element.text + "\n"
                
                # Ajoute le texte à l'historique pour contexte des images/tableaux suivants
                text_list.append(element.text)

        # Wrap le texte combiné dans un objet Document LangChain
        # Cet objet sera utilisé par le splitter pour découper en chunks
        document = Document(page_content=text)

        # Découpe le document en chunks cohérents avec chevauchement
        # chunk_size=1000: Taille cible de chaque chunk (caractères)
        # chunk_overlap=200: Chevauchement pour préserver contexte entre chunks
        # Applique la division récursive (partage par sections, paragraphes, phrases)
        chunks = text_splitter.split_documents([document])
        
        # Retourne la liste des chunks prêts pour l'embedding FAISS
        return chunks


# ============================================================================
# PROGRAMME PRINCIPAL: EXEMPLE D'UTILISATION
# ============================================================================
if __name__ == "__main__":
    """
    Exemple d'utilisation du MultimodalEmbedder pour traiter un PDF complexe.
    
    ÉTAPES:
    -------
    1. Crée une instance de l'embedder multimodal
    2. Charge et traite un PDF (extraction, résumé, splitting)
    3. Embed les chunks dans FAISS avec ID de document
    4. Effectue une recherche sémantique simple
    """
    # TODO: Implémenter exemple avec clé API
    # embedder = MultimodalEmbedder(api_key="votre-clé-api")
    # chunks = embedder.load_and_split("chemin/vers/document.pdf")
    # embedder.embed(chunks, doc_id="mon_document")
    pass

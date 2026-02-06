# ============================================================================
# IMPORTS
# ============================================================================
# RecursiveCharacterTextSplitter: Outil de division de texte par caractères récursifs
from langchain.text_splitter import RecursiveCharacterTextSplitter
# PyPDFLoader: Chargeur LangChain pour extraire texte natif des PDFs
from langchain_community.document_loaders import PyPDFLoader
# Embedder: Classe parent fournissant les fonctionnalités de base d'embeddings FAISS
from Embedder import Embedder

# ============================================================================
# CLASSE TEXTEMBEDDER
# ============================================================================
class TextEmbedder(Embedder):
    """
    Embedder simple pour traiter des PDFs contenant principalement du texte natif.
    
    RESPONSABILITÉS:
    ----------------
    - Chargement de documents PDF avec extraction de texte natif
    - Découpage intelligent en chunks cohérents (taille 1000, chevauchement 200)
    - Préparation des chunks pour l'embedding FAISS
    
    UTILISATION:
    -----------
    - Documents textuels simples: Rapports, articles, livres
    - Archivage et indexation: Vectorisation pour recherche sémantique (RAG)
    - Cas d'usage: Lorsque le PDF contient du texte natif (pas d'images/tableaux)
    
    COMPARAISON AVEC AUTRES EMBEDDERS:
    ----------------------------------
    - TextEmbedder: Texte natif uniquement (simple, rapide)
    - EmbedderWithOcr: Texte natif + OCR pour images (intermédiaire)
    - MultimodalEmbedder: Texte + Images (vision) + Tableaux (LLM) (avancé)
    
    ATTRIBUTS HÉRITÉS:
    -----------------
    - api_key: Clé API Mistral pour l'embedding FAISS
    - faiss_path: Chemin vers l'index FAISS
    - document_index: Dictionnaire UUID -> métadonnées document
    
    NOTE: Cette classe hérite de Embedder et fournit les fonctionnalités
          minimales pour traiter des PDFs textuels standards.
    """
    
    def __init__(self, api_key):
        """
        Initialise l'embedder de texte simple.
        
        PROCESSUS:
        ----------
        1. Appelle le constructeur parent (Embedder) pour initialiser:
           - La clé API Mistral
           - Le chemin FAISS
           - L'index de documents
        
        ARGS:
        ----
        api_key (str): Clé API Mistral pour l'embedding des chunks
        
        EXEMPLE:
        -------
        >>> embedder = TextEmbedder(api_key="your-api-key")
        """
        # Appel du constructeur parent pour initialiser la clé API et les chemins FAISS
        super().__init__(api_key)
    
    # ========================================================================
    # MÉTHODE DE CHARGEMENT ET SPLITTING
    # ========================================================================
    def load_and_split(self, pdf_path):
        """
        Charge un PDF et le découpe en chunks cohérents pour l'embedding.
        
        PROCESSUS:
        ----------
        1. Initialise le chargeur LangChain PyPDFLoader
        2. Charge le document PDF (extraction texte natif uniquement)
        3. Initialise un splitter de texte (chunk_size=1000, overlap=200)
        4. Découpe le document en chunks cohérents et chevauchants
        5. Retourne la liste des chunks prêts pour l'embedding FAISS
        
        ARGS:
        ----
        pdf_path (str): Chemin absolu du fichier PDF à traiter
        
        RETURNS:
        -------
        list[Document]: Liste de chunks LangChain prêts pour l'embedding
        
        SPLITTING STRATEGY:
        -------------------
        - Taille de chunk: 1000 caractères
        - Chevauchement: 200 caractères (pour préserver contexte entre chunks)
        - Stratégie: Récursive (divise par sections, paragraphes, phrases)
        - Bénéfice: Les chunks se chevauchent pour une meilleure récupération sémantique
        
        NOTES:
        -----
        - PyPDFLoader extrait uniquement le texte natif (pas OCR)
        - Les images et tableaux sont ignorés
        - La taille 1000 + overlap 200 équilibre contexte et granularité
        - Cette approche est appropriée pour des PDFs textuels simples
        
        EXEMPLE:
        -------
        >>> embedder = TextEmbedder(api_key="...")
        >>> chunks = embedder.load_and_split("/chemin/to/document.pdf")
        >>> print(f"Nombre de chunks: {len(chunks)}")
        >>> for chunk in chunks[:3]:
        >>>     print(chunk.page_content[:100])  # Aperçu du contenu
        >>> # Ensuite, utiliser chunks pour embedding FAISS:
        >>> # embedder.embed(chunks, doc_id="mon_doc")
        """
        # Initialise le chargeur PyPDFLoader pour extraire le texte natif du PDF
        loader = PyPDFLoader(pdf_path)
        
        # Charge le PDF et retourne une liste de Documents LangChain
        # Chaque document représente une page du PDF avec son texte
        documents = loader.load()
        
        # Initialise le splitter avec taille=1000 et chevauchement=200
        # Cela assure une bonne granularité tout en gardant du contexte
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # Découpe les documents chargés en chunks cohérents et chevauchants
        # La stratégie récursive divise d'abord par sections, puis paragraphes, puis phrases
        chunks = text_splitter.split_documents(documents)
        
        # Retourne la liste des chunks prêts pour l'embedding FAISS
        return chunks


# ============================================================================
# PROGRAMME PRINCIPAL: EXEMPLE D'UTILISATION
# ============================================================================
if __name__ == "__main__":
    """
    Exemple d'utilisation du TextEmbedder pour traiter un PDF textuel simple.
    
    ÉTAPES:
    -------
    1. Crée une instance de l'embedder de texte
    2. Charge et traite un PDF (extraction texte, splitting)
    3. Embed les chunks dans FAISS avec ID de document
    """
    # TODO: Implémenter exemple avec clé API
    # embedder = TextEmbedder(api_key="votre-clé-api")
    # chunks = embedder.load_and_split("chemin/vers/document.pdf")
    # embedder.embed(chunks, doc_id="mon_document")
    pass
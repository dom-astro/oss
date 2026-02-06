# ==============================================================================
# IMPORTS - Dépendances pour l'embedding et la vectorisation de documents
# ==============================================================================

# Embeddings Mistral - Transformation de texte en vecteurs
from langchain_mistralai.embeddings import MistralAIEmbeddings

# FAISS - Base de données vectorielle pour recherche sémantique
from langchain_community.vectorstores import FAISS

# Utilitaires Python
import os  # Gestion des fichiers/répertoires
from dotenv import load_dotenv  # Chargement des variables d'environnement
from pathlib import Path  # Gestion des chemins fichiers
import json  # Manipulation de JSON
import uuid  # Génération d'identifiants uniques

# ==============================================================================
# CLASSE EMBEDDER - Gestion de base des embeddings
# ==============================================================================
# Cette classe gère la création et la gestion des embeddings de documents PDF


class Embedder:
    """
    Classe pour gérer les embeddings de documents PDF et leur stockage en base vectorielle FAISS.
    
    Responsabilités:
    - Charger et découper les PDF
    - Créer des embeddings avec le modèle Mistral
    - Stocker/récupérer les vecteurs dans FAISS
    - Tracker les documents par identifiant unique
    - Supprimer des documents de l'index
    """
    
    def __init__(self, api_key):
        """
        Initialise l'embedder avec la clé API Mistral.
        
        Args:
            api_key (str): Clé API Mistral pour l'authentification
        """
        self.api_key = api_key
        # Chemin vers le répertoire contenant l'index FAISS
        self.faiss_index_path = Path(__file__).resolve().parent.parent / "faiss_index"
        # Chemin vers le fichier de tracking des documents (nom -> UUID)
        self.document_index_path = Path(__file__).resolve().parent.parent / "document_index.json"
        
        # Charger le mapping document ↔ doc_id s'il existe
        if self.document_index_path.exists():
            with open(self.document_index_path, "r") as f:
                self.document_index = json.load(f)
        else:
            self.document_index = {}
    
    def load_and_split(self, pdf_path):
        """
        Charge et découpe un fichier PDF en chunks.
        
        À implémenter dans les classes dérivées (TextEmbedder, EmbedderWithOcr, etc.).
        
        Args:
            pdf_path (str): Chemin vers le fichier PDF
            
        Returns:
            list: Liste de documents (chunks) avec métadonnées
        """
        pass 
    def embed(self, pdf_path, store, save=False):
        """
        Crée des embeddings à partir du contenu d'un PDF et les stocke dans l'index FAISS.
        
        Processus:
        1. Génère un UUID unique pour le document
        2. Découpe le PDF en chunks
        3. Ajoute le doc_id aux métadonnées de chaque chunk
        4. Crée ou augmente l'index FAISS
        5. Optionnellement: sauvegarde l'index sur disque
        
        Args:
            pdf_path (str): Chemin vers le fichier PDF à traiter
            store (FAISS or None): Index FAISS existant, ou None pour en créer un nouveau
            save (bool): Si True, sauvegarde l'index sur disque et met à jour document_index.json
        
        Returns:
            FAISS: L'index vectoriel mis à jour
        """
        # Génère un identifiant unique pour ce document
        doc_id = str(uuid.uuid4())

        # Charge et découpe le PDF en chunks
        chunks = self.load_and_split(pdf_path)

        # Ajoute le doc_id dans les métadonnées pour chaque chunk
        # Cela permet de tracker quel vecteur appartient à quel document
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
        
        # Crée un nouvel index ou augmente l'existant
        if store is None:
            # Crée un nouvel index vectoriel à partir des chunks
            embeddings = MistralAIEmbeddings(
                model="mistral-embed",
                mistral_api_key=self.api_key,
            )
            store = FAISS.from_documents(chunks, embeddings)
        else:
            # Ajoute les chunks au nouvel index existant
            store.add_documents(chunks)
        
        # Optionnellement sauvegarde l'index et le mapping
        if save:
            # Sauvegarde l'index vectoriel FAISS sur disque
            store.save_local(str(self.faiss_index_path))
            # Enregistre la correspondance nom_fichier ↔ doc_id
            self.document_index[os.path.basename(pdf_path)] = doc_id
            with open(self.document_index_path, "w", encoding="utf-8") as f:
                json.dump(self.document_index, f, indent=2)
        
        return store


    def delete_document(self, doc_name: str, store, save=False) -> bool:
        """
        Supprime tous les vecteurs appartenant à un document de l'index FAISS.
        
        N'effectue pas de ré-embedding, supprime simplement les vecteurs existants
        en temps constant dans FAISS.
        
        Processus:
        1. Trouve le doc_id associé au nom du document
        2. Identifie tous les vecteurs avec ce doc_id
        3. Supprime les vecteurs de l'index FAISS
        4. Optionnellement: sauvegarde et met à jour le tracking
        
        Args:
            doc_name (str): Nom du fichier du document à supprimer (ex: "rapport.pdf")
            store (FAISS): L'index vectoriel
            save (bool): Si True, sauvegarde les modifications sur disque
        
        Returns:
            bool: True si suppression réussie, False si le document n'existe pas
        """
        # Vérifie que le document existe dans le tracking
        if doc_name not in self.document_index or store is None:
            return False
        
        # Récupère l'UUID unique associé à ce document
        target_doc_id = self.document_index[doc_name]

        # Collecte tous les IDs de vecteurs à supprimer
        # Parcourt le mapping index_to_docstore_id pour trouver les vecteurs avec le bon doc_id
        ids_to_delete = [
            ds_id
            for row, ds_id in store.index_to_docstore_id.items()
            if store.docstore.search(ds_id).metadata.get("doc_id") == target_doc_id
        ]
        
        # Si aucun vecteur ne correspond, le document est déjà supprimé
        if not ids_to_delete:
            return True

        # Supprime les vecteurs de l'index FAISS (opération en temps constant)
        store.delete(ids=ids_to_delete)
        
        # Optionnellement sauvegarde les modifications
        if save:
            # Sauvegarde l'index FAISS mis à jour sur disque
            store.save_local(str(self.faiss_index_path))
            # Supprime l'entrée du tracking document ↔ doc_id
            del self.document_index[doc_name]
            with open(self.document_index_path, "w") as f:
                json.dump(self.document_index, f, indent=2)
        
        return True

    def vectors_for_pdf(self, doc_id=None) -> int | set:
        """
        Retourne le nombre de vecteurs ou les IDs uniques pour un document.
        
        Si doc_id est fourni: retourne le nombre de vecteurs pour ce document.
        Si doc_id est None: retourne l'ensemble de tous les doc_id présents.
        
        Comportement:
        - Charge l'index FAISS depuis le disque
        - Cherche les vecteurs avec le doc_id spécifié
        - Retourne le nombre ou l'ensemble selon les paramètres
        
        Args:
            doc_id (str or None): UUID du document, ou None pour tous les docs
        
        Returns:
            int: Nombre de vecteurs (si doc_id fourni)
            set: Ensemble des doc_id uniques (si doc_id is None)
            0: Si aucun vecteur ne correspond
        """
        # Charge les embeddings pour accéder à l'index
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=self.api_key,
        )

        # Charge l'index FAISS depuis le disque
        store = FAISS.load_local(
            str(self.faiss_index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        
        # Cas 1: Rechercher les vecteurs pour un doc_id spécifique
        if doc_id is not None:
            target_doc_id = doc_id
            # Compte les vecteurs avec ce doc_id
            ids = [
                ds_id
                for ds_id in store.index_to_docstore_id.values()
                if store.docstore.search(ds_id).metadata.get("doc_id") == target_doc_id
            ]
            return len(ids)
        
        # Cas 2: Récupérer l'ensemble de tous les doc_id présents
        else:
            # Récupère tous les IDs de docstore
            available_ids = [
                ds_id
                for ds_id in store.index_to_docstore_id.values()
            ]
            # Extrait les doc_id uniques de chaque vecteur
            available_doc_ids = [
                store.docstore.search(ds_id).metadata.get("doc_id")
                for ds_id in available_ids
            ]
            # Retourne l'ensemble (élimine les doublons)
            return set(available_doc_ids)
# ==============================================================================
# PROGRAMME PRINCIPAL - Test et démonstration
# ==============================================================================

if __name__ == "__main__":
    # Charge les variables d'environnement
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    load_dotenv()
    
    # Chemins vers les répertoires de travail
    DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
    api_key = os.getenv("MISTRAL_API_KEY")
    
    # Crée une instance d'Embedder et affiche les doc_id disponibles
    embedder = Embedder(api_key)
    print(embedder.vectors_for_pdf())
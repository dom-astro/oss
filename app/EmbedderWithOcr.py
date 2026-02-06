# ==============================================================================
# IMPORTS - Dépendances pour l'extraction OCR et le traitement de PDF
# ==============================================================================

# LangChain - Documents et text splitters
from langchain.schema import Document  # Format de document standard
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Découpe intelligente

# PDF et Image Processing
import fitz  # PyMuPDF - Extraction de texte et images depuis PDF
import pytesseract  # OCR (Optical Character Recognition) pour extraire texte des images
from PIL import Image  # Traitement d'images
from pdf2image import convert_from_path  # Conversion PDF → images

# Utilitaires
import io  # Gestion de flux de bytes

# Embedder parent
from Embedder import Embedder  # Classe de base pour l'embedding

# ==============================================================================
# CLASSE EmbedderWithOcr - Embedding avec OCR pour extraire du texte d'images
# ==============================================================================
# Cette classe étend Embedder pour supporter les PDF avec beaucoup d'images

class EmbedderWithOcr(Embedder):
    """
    Embedder avancé qui extrait du texte ET des images des PDF, en utilisant OCR.
    
    Hérite de la classe Embedder et en surcharge les méthodes d'extraction.
    
    Caractéristiques:
    - Extrait le texte natif du PDF (si disponible)
    - Extrait les images intégrées dans le PDF
    - Effectue une reconnaissance optique de caractères (OCR) sur les images
    - Fournit un mécanisme de secours (fallback) pour les pages sans texte
    - Combine le texte natif et l'OCR en un contenu unifié
    
    Cas d'usage idéaux:
    - Documents scannés avec du texte ET des images
    - Rapports avec diagrammes/graphiques importants
    - Documents avec beaucoup de contenu visuel
    """
    
    def __init__(self, api_key):
        """
        Initialise l'embedder OCR avec la clé API Mistral.
        
        Args:
            api_key (str): Clé API Mistral pour l'authentification
        """
        super().__init__(api_key)

    def extract_text_and_images(self, pdf_path):
        """
        Extrait du texte ET des images d'un fichier PDF, en appliquant l'OCR si nécessaire.
        
        Processus pour chaque page:
        1. Extrait le texte natif du PDF (s'il existe)
        2. Extrait les images intégrées
        3. Applique l'OCR sur chaque image
        4. Combine le texte natif + OCR
        5. Si aucun texte: utilise une méthode de secours (convertir page en image + OCR)
        
        Args:
            pdf_path (str): Chemin vers le fichier PDF
        
        Returns:
            list: Liste de Document objects contenant le texte extrait et les métadonnées
        """
        # Ouvre le fichier PDF
        doc = fitz.open(pdf_path)
        documents = []
        processed_xrefs = set()  # Trackage des références d'images pour éviter les doublons
        
        # Parcourt chaque page du PDF
        for i, page in enumerate(doc):
            print(f"Traitement de la page {i+1}/{len(doc)}")
            
            # Extrait le texte directement depuis le PDF
            page_text = page.get_text() or ""
            image_texts = []
            
            # Extrait les images présentes sur la page
            page_images = page.get_images(full=True)
            print(f"  Trouvé {len(page_images)} images sur la page {i+1}")

            # Traite chaque image trouvée
            for img_index, img_info in enumerate(page_images):
                xref = img_info[0]  # Référence de l'image dans le PDF

                # Ignore si nous avons déjà traité cette image
                if xref in processed_xrefs:
                    continue
                processed_xrefs.add(xref)

                try:
                    # Extrait les bytes de l'image et convertit en PIL Image
                    img = doc.extract_image(xref)
                    img_bytes = img["image"]
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                    # Applique l'OCR pour extraire le texte de l'image
                    extracted_text = pytesseract.image_to_string(pil_img)
                    if extracted_text.strip():
                        image_texts.append(extracted_text)

                except Exception as e:
                    print(f"  [ATTENTION] Impossible de traiter l'image xref={xref}: {e}")
                    continue

            # Combine le texte natif du PDF avec les résultats de l'OCR
            combined_text = (page_text + "\n" + "\n".join(image_texts)).strip()

            # Plan de secours: Si aucun texte n'a été extrait, convertir la page entière en image et OCR
            if not combined_text:
                try:
                    # Convertit la page en image (150 DPI)
                    fallback_image = convert_from_path(pdf_path, first_page=i+1, last_page=i+1, dpi=150)[0]

                    # Applique l'OCR sur la page entière
                    fallback_text = pytesseract.image_to_string(fallback_image)
                    if fallback_text.strip():
                        combined_text = fallback_text.strip()
                except Exception as e:
                    print(f"  [Plan de secours échoué] Page {i+1}: {e}")

            # Crée un Document si du texte a été extrait
            if combined_text:
                documents.append(
                    Document(
                        page_content=combined_text,
                        metadata={
                            "source": pdf_path,
                            "page": i + 1,
                        }
                    )
                )

        return documents


    def load_and_split(self, pdf_path):
        """
        Charge un PDF, extrait le texte et les images, et découpe en chunks pour embedding.
        
        Processus:
        1. Extrait texte natif + OCR des images via extract_text_and_images()
        2. Découpe les documents en chunks de 1000 caractères (avec 200 caractères de chevauchement)
        3. Retourne les chunks prêts pour l'embedding
        
        Args:
            pdf_path (str): Chemin vers le fichier PDF
        
        Returns:
            list: Liste de Document chunks prêts pour l'embedding
        """
        # Extrait le texte et les images du PDF
        raw_docs = self.extract_text_and_images(pdf_path)
        
        # Découpe les documents en chunks plus petits pour un meilleur embedding
        # Utilise le découpage récursif pour préserver la structure du texte
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(raw_docs)
        
        return chunks

# ==============================================================================
# PROGRAMME PRINCIPAL - Test et démonstration
# ==============================================================================

if __name__ == "__main__":
    # Charge les variables d'environnement
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    load_dotenv()
    
    # Définit les répertoires de travail
    DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
    api_key = os.getenv("MISTRAL_API_KEY")
    
    # Crée une instance d'EmbedderWithOcr et traite un PDF exemple
    embedder = EmbedderWithOcr(api_key)
    save_path = os.path.join(DOCS_DIR, "astro-procedures-resume-anon.pdf")
    embedder.embed(save_path)
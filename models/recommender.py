import re
import logging
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize

# 1. Configuration d'un logger professionnel (fini les print silencieux)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """
    Service d'analyse sémantique et d'extraction de points clés.
    Implémente le pattern Singleton (implicite via l'instanciation) pour éviter
    de recharger le modèle IA de 400Mo à chaque requête.
    """

    def __init__(self, model_name="paraphrase-multilingual-mpnet-base-v2"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initialisation de DocumentAnalyzer sur : {self.device.upper()}")

        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self._ensure_nltk_resources()
            logger.info("Modèle IA et ressources NLP chargés avec succès.")
        except Exception as e:
            logger.error(f"Erreur fatale lors du chargement de l'IA : {str(e)}")
            raise

    def _ensure_nltk_resources(self):
        """Vérifie et télécharge les modèles de ponctuation si manquants."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Téléchargement du tokenizer NLTK 'punkt'...")
            nltk.download('punkt', quiet=True)

    def _clean_pdf_artifacts(self, text: str) -> str:
        """ Nettoyage chirurgical spécialisé pour les extractions PDF. """
        if not text:
            return ""

        # Uniformisation des espaces et apostrophes
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = text.replace('’', "'").replace('`', "'").replace('«', '"').replace('»', '"')

        # Fix 1: Mots coupés par un tiret (ex: "deman- der" -> "demander")
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

        # Fix 2: Apostrophes décollées (ex: "l' avocat" ou "l ' avocat" -> "l'avocat")
        text = re.sub(r"\b([ldjmtscnLDJMTSCN])\s*[']\s+(\w)", r"\1'\2", text)

        # Fix 3: Lettres isolées au milieu d'un mot (ex: "h istoire" -> "histoire")
        def fix_isolated(match):
            char = match.group(1).lower()
            if char in ['a', 'y', 'l', 'd', 'j', 'm', 't', 's', 'c', 'n']:
                return match.group(0)  # Mot légitime d'une lettre
            return match.group(1) + match.group(2)

        text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z]{2,})\b', fix_isolated, text)

        # Fix 4: Espaces multiples
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _format_output(self, sentence: str) -> str:
        """ Post-traitement cosmétique pour l'interface utilisateur. """
        sentence = sentence.strip()
        if not sentence:
            return ""
        # Majuscule forcée au début
        sentence = sentence[0].upper() + sentence[1:]
        # Ponctuation forcée à la fin si absente
        if not sentence.endswith(('.', '!', '?', '"')):
            sentence += '.'
        return sentence

    def extract_keypoints(self, text: str, top_k: int = 5, diversity: float = 0.35) -> list:
        """
        Extrait les points clés en utilisant Sentence-BERT et MMR.
        """
        cleaned_text = self._clean_pdf_artifacts(text)

        if len(cleaned_text) < 50:
            logger.warning("Texte trop court pour l'extraction.")
            return []

        # Tokenization robuste
        try:
            raw_sentences = sent_tokenize(cleaned_text, language='french')
        except Exception as e:
            logger.warning(f"Échec NLTK, fallback sur Regex. Erreur: {e}")
            raw_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', cleaned_text)

        # Filtrage heuristique strict
        valid_sentences = []
        for s in raw_sentences:
            s_clean = s.strip()
            # On rejette: Trop court, trop long, dialogues, ou commençant par une minuscule (souvent un fragment PDF)
            if (40 <= len(s_clean) <= 300 and
                    not s_clean.startswith(('"', '-', '—', '•', '*')) and
                    s_clean[0].isupper()):
                valid_sentences.append(s_clean)

        if len(valid_sentences) <= top_k:
            return [self._format_output(s) for s in valid_sentences]

        # Encodage vectoriel
        embeddings = self.model.encode(valid_sentences, convert_to_tensor=True)
        cos_sim_matrix = util.cos_sim(embeddings, embeddings)
        doc_centrality = cos_sim_matrix.sum(dim=1).cpu().numpy()

        # Algorithme MMR
        selected_indices = []
        unselected_indices = list(range(len(valid_sentences)))

        first_idx = int(np.argmax(doc_centrality))
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)

        max_centrality = doc_centrality[first_idx]

        while len(selected_indices) < top_k and unselected_indices:
            best_mmr_score = -np.inf
            best_idx = -1

            for idx in unselected_indices:
                relevance = doc_centrality[idx]
                redundancy = max([cos_sim_matrix[idx][s_idx].item() for s_idx in selected_indices])

                mmr_score = ((1 - diversity) * relevance) - (diversity * redundancy * max_centrality)

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

        # Tri chronologique pour la cohérence de lecture
        selected_indices.sort()

        return [self._format_output(valid_sentences[i]) for i in selected_indices]

# ==========================================
# UTILISATION (Exemple pour ton contrôleur)
# ==========================================
# Instancier le service UNE SEULE FOIS au démarrage de l'application/du worker
# analyzer = DocumentAnalyzer()
#
# Lors d'une requête web :
# resultats = analyzer.extract_keypoints(mon_texte_pdf, top_k=5)
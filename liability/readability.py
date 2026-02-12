"""Readability metrics computation for medical recommendations."""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np

# Lazy imports (only load when needed)
_transformers_available = False
_spacy_available = False
_sentence_transformer_available = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _transformers_available = True
except ImportError:
    pass

try:
    import spacy
    _spacy_available = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    _sentence_transformer_available = True
except ImportError:
    pass

import textstat  # Fallback for traditional metrics


class ReadabilityAnalyzer:
    """Compute readability metrics for medical recommendations."""

    def __init__(
        self,
        transformer_model: str = "agentlans/mdeberta-v3-base-readability",
        sentence_model: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False,
    ):
        """
        Initialize readability analyzer.

        Args:
            transformer_model: HuggingFace model for readability scoring
            sentence_model: Model for sentence embeddings (coherence)
            use_gpu: Whether to use GPU acceleration
        """
        self.transformer_model_name = transformer_model
        self.sentence_model_name = sentence_model

        # Determine device
        if use_gpu and _transformers_available:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if not torch.cuda.is_available() and use_gpu:
                warnings.warn("GPU requested but CUDA not available, using CPU")
        else:
            self.device = "cpu"

        # Lazy loading of models (only when actually used)
        self._transformer_model = None
        self._transformer_tokenizer = None
        self._sentence_model = None
        self._spacy_nlp = None

    def _load_transformer_model(self):
        """Lazy load transformer model."""
        if not _transformers_available:
            warnings.warn("transformers not available, skipping transformer metrics")
            return False

        if self._transformer_model is None:
            try:
                self._transformer_tokenizer = AutoTokenizer.from_pretrained(
                    self.transformer_model_name
                )
                self._transformer_model = AutoModelForSequenceClassification.from_pretrained(
                    self.transformer_model_name
                ).to(self.device)
                self._transformer_model.eval()
            except Exception as e:
                warnings.warn(f"Failed to load transformer model: {e}")
                return False
        return True

    def _load_sentence_model(self):
        """Lazy load sentence embedding model."""
        if not _sentence_transformer_available:
            warnings.warn("sentence-transformers not available, skipping coherence metrics")
            return False

        if self._sentence_model is None:
            try:
                self._sentence_model = SentenceTransformer(
                    self.sentence_model_name,
                    device=self.device
                )
            except Exception as e:
                warnings.warn(f"Failed to load sentence model: {e}")
                return False
        return True

    def _load_spacy(self):
        """Lazy load spaCy."""
        if not _spacy_available:
            warnings.warn("spacy not available, skipping pronoun density")
            return False

        if self._spacy_nlp is None:
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                warnings.warn("spacy model not found. Run: python -m spacy download en_core_web_sm")
                return False
        return True

    def compute_traditional_metrics(self, text: str) -> dict[str, float | None]:
        """
        Compute traditional readability metrics.

        Args:
            text: The recommendation text

        Returns:
            Dictionary with flesch_kincaid_grade and smog_index
        """
        if not text or len(text) < 50:
            return {"flesch_kincaid_grade": None, "smog_index": None}

        try:
            fk_grade = textstat.flesch_kincaid_grade(text)
            smog = textstat.smog_index(text)

            # Bound to reasonable ranges
            fk_grade = max(0.0, min(20.0, fk_grade))
            smog = max(0.0, min(20.0, smog))

            return {
                "flesch_kincaid_grade": round(fk_grade, 2),
                "smog_index": round(smog, 2),
            }
        except Exception as e:
            warnings.warn(f"Traditional metrics failed: {e}")
            return {"flesch_kincaid_grade": None, "smog_index": None}

    def compute_transformer_readability(self, text: str) -> float | None:
        """
        Compute transformer-based readability score.

        Args:
            text: The recommendation text

        Returns:
            Readability score (interpretation depends on model)
        """
        if not text or len(text) < 50:
            return None

        if not self._load_transformer_model():
            return None

        try:
            # Truncate to model's max length
            inputs = self._transformer_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self._transformer_model(**inputs)
                # For regression models, take the logit value
                # For classification, take predicted class
                if outputs.logits.shape[-1] == 1:
                    # Regression
                    score = outputs.logits.item()
                else:
                    # Classification (predict grade level)
                    score = outputs.logits.argmax(-1).item()

            return round(float(score), 2)

        except Exception as e:
            warnings.warn(f"Transformer readability failed: {e}")
            return None

    def compute_discourse_cohesion(self, text: str) -> dict[str, float | None]:
        """
        Compute discourse cohesion metrics.

        Args:
            text: The recommendation text

        Returns:
            Dictionary with cohesion metrics
        """
        if not text or len(text) < 50:
            return {
                "lexical_overlap_adjacent": None,
                "lexical_overlap_global": None,
                "pronoun_density": None,
                "semantic_coherence_local": None,
                "semantic_coherence_global": None,
            }

        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return {
                "lexical_overlap_adjacent": None,
                "lexical_overlap_global": None,
                "pronoun_density": None,
                "semantic_coherence_local": None,
                "semantic_coherence_global": None,
            }

        # Compute individual metrics
        lexical_adjacent = self._lexical_overlap_adjacent(sentences)
        lexical_global = self._lexical_overlap_global(sentences)
        pronoun_dens = self._pronoun_density(text)
        semantic_local = self._semantic_coherence_local(sentences)
        semantic_global = self._semantic_coherence_global(sentences)

        return {
            "lexical_overlap_adjacent": lexical_adjacent,
            "lexical_overlap_global": lexical_global,
            "pronoun_density": pronoun_dens,
            "semantic_coherence_local": semantic_local,
            "semantic_coherence_global": semantic_global,
        }

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitter (can be enhanced with spaCy)
        # Split on . ! ? followed by space and capital letter
        sentences = re.split(r'[.!?]+\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _lexical_overlap_adjacent(self, sentences: list[str]) -> float | None:
        """Jaccard similarity between adjacent sentences."""
        try:
            overlaps = []
            for i in range(len(sentences) - 1):
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i + 1].lower().split())

                if not words1 or not words2:
                    continue

                intersection = len(words1 & words2)
                union = len(words1 | words2)

                if union > 0:
                    overlaps.append(intersection / union)

            return round(float(np.mean(overlaps)), 3) if overlaps else None
        except Exception:
            return None

    def _lexical_overlap_global(self, sentences: list[str]) -> float | None:
        """Global lexical cohesion across all sentence pairs."""
        try:
            # Count content words appearing in multiple sentences
            word_sentence_map = {}
            for i, sent in enumerate(sentences):
                words = set(sent.lower().split())
                for word in words:
                    if len(word) > 3:  # Filter short words
                        word_sentence_map.setdefault(word, set()).add(i)

            # Calculate proportion of words appearing in 2+ sentences
            multi_sentence_words = sum(
                1 for sents in word_sentence_map.values() if len(sents) > 1
            )
            total_unique_words = len(word_sentence_map)

            if total_unique_words == 0:
                return None

            return round(multi_sentence_words / total_unique_words, 3)
        except Exception:
            return None

    def _pronoun_density(self, text: str) -> float | None:
        """Pronouns per 100 words (referential cohesion proxy)."""
        if not self._load_spacy():
            # Fallback: regex-based pronoun counting
            pronouns = re.findall(
                r'\b(I|you|he|she|it|we|they|me|him|her|us|them|my|your|his|its|our|their)\b',
                text,
                re.IGNORECASE
            )
            words = text.split()
            if not words:
                return None
            return round((len(pronouns) / len(words)) * 100, 2)

        try:
            doc = self._spacy_nlp(text)
            pronouns = [token for token in doc if token.pos_ == "PRON"]
            total_words = len([token for token in doc if token.is_alpha])

            if total_words == 0:
                return None

            return round((len(pronouns) / total_words) * 100, 2)
        except Exception:
            return None

    def _semantic_coherence_local(self, sentences: list[str]) -> float | None:
        """Average cosine similarity between adjacent sentences."""
        if not self._load_sentence_model():
            return None

        try:
            embeddings = self._sentence_model.encode(
                sentences,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            similarities = []
            for i in range(len(embeddings) - 1):
                # Cosine similarity
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                similarities.append(sim)

            return round(float(np.mean(similarities)), 3) if similarities else None
        except Exception:
            return None

    def _semantic_coherence_global(self, sentences: list[str]) -> float | None:
        """Average cosine similarity across all sentence pairs."""
        if not self._load_sentence_model():
            return None

        try:
            embeddings = self._sentence_model.encode(
                sentences,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # Compute pairwise similarities
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(embeddings)

            # Take upper triangle (excluding diagonal)
            n = len(sim_matrix)
            if n < 2:
                return None

            upper_tri = sim_matrix[np.triu_indices(n, k=1)]
            return round(float(np.mean(upper_tri)), 3)
        except Exception:
            return None

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Compute all readability metrics.

        Args:
            text: The recommendation text

        Returns:
            Dictionary with all computed metrics
        """
        results = {}

        # Traditional metrics (always computed, fast)
        traditional = self.compute_traditional_metrics(text)
        results.update(traditional)

        # Transformer-based (slower, optional)
        results["transformer_readability_score"] = self.compute_transformer_readability(text)
        results["transformer_model_name"] = (
            self.transformer_model_name if results["transformer_readability_score"] is not None else None
        )

        # Discourse cohesion metrics
        cohesion = self.compute_discourse_cohesion(text)
        results.update(cohesion)

        return results


# Singleton instance for batch processing
_analyzer_instance: ReadabilityAnalyzer | None = None

def get_analyzer(use_gpu: bool = False) -> ReadabilityAnalyzer:
    """Get or create singleton analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ReadabilityAnalyzer(use_gpu=use_gpu)
    return _analyzer_instance

from typing import Tuple

from sentence_transformers import CrossEncoder


class RAGVerificationManager:

    def __init__(self):
        # Init cross encoder
        self._rag_verification_model_ = CrossEncoder(
            "cross-encoder/nli-deberta-v3-base")

    def verify_RAG_completion(self,
                              completion: str,
                              context_chunks: list,
                              entailment_threshold: float
                              ) -> Tuple[bool, str, list]:
        # Divide completion in sentences
        sentences = completion.split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        # Create pairs for chunks/sentences & get scores
        # Chunks are the premise and sentence the hypotheses
        pairs = [(context, sentence) for sentence in sentences
                 for context in context_chunks]
        scores = self._rag_verification_model_.predict(
            pairs, apply_softmax=True)

        # Get entailed & contradiction index
        entail_idx = self._rag_verification_model_.config.label2id[
            "entailment"]

        # Group scores by sentence
        entail_scores = [
            [
                scores[i * len(context_chunks) + j][entail_idx]
                for j in range(len(context_chunks))
            ]
            for i in range(len(sentences))
        ]

        # Filter sentences (at least one score above threshold)
        accepted_sentences = []
        refused_sentences = []

        for s, s_scores in zip(sentences, entail_scores):
            if any(score >=
                    entailment_threshold
                    for score in s_scores):
                accepted_sentences.append(s)
            else:
                refused_sentences.append(s)

        # Merge in completion & fill removed claims
        verified_completion = " ".join(accepted_sentences)

        return True, verified_completion, refused_sentences

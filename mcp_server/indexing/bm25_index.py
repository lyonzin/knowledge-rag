"""
╭─╴ BM25 INVERTED INDEX ╶────────────────────────────────────────╮
│                                                                │
│   Custom BM25Okapi implementation with inverted-index          │
│   acceleration — scores only documents containing query        │
│   terms instead of scanning the full corpus.                   │
│                                                                │
│   Extracted verbatim from server.py in the A2.1 refactor.      │
│                                                                │
╰────────────────────────────────────────────────────────────────╯

    ┌─ Author  ·  Ailton Rocha (Lyon.)
    └─ Version ·  single-sourced from ``mcp_server.__version__``
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import config
from ..stopwords import filter_query_stopwords


class BM25Index:
    """
    BM25 keyword index with inverted-index acceleration for hybrid search.

    Uses a custom inverted index to score only documents containing query terms
    instead of scanning the entire corpus. Produces scores identical to BM25Okapi
    (k1=1.5, b=0.75) but runs in O(matching_docs) instead of O(corpus_size).
    """

    def __init__(self):
        self.corpus: List[str] = []
        self.corpus_ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self._idf: Dict[str, float] = {}
        self._doc_len: Optional[np.ndarray] = None
        self._avgdl: float = 0.0
        self._corpus_size: int = 0
        self._k1: float = 1.5
        self._b: float = 0.75
        self._epsilon: float = 0.25
        self._index_built: bool = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric, keep hyphens"""
        text_lower = text.lower()
        tokens = re.findall(r"[a-z0-9][-a-z0-9]*[a-z0-9]|[a-z0-9]", text_lower)
        return tokens

    def expand_query(self, query: str) -> str:
        """
        Expand query with configured synonyms for BM25 search.

        Looks up full-query, token, and bigram matches against the merged
        expansion table from config. Improves keyword recall for abbreviated
        and synonymous technical terms (e.g., "sqli" expands to include
        "sql injection").

        Args:
            query: Original query string

        Returns:
            Expanded query string with synonyms appended
        """
        query_lower = query.lower().strip()
        expansions = config.query_expansions
        expanded_terms: List[str] = []
        seen_terms = set()
        seen_add = seen_terms.add
        expanded_append = expanded_terms.append

        # Check full query
        full_query_terms = expansions.get(query_lower)
        if full_query_terms:
            for term in full_query_terms:
                if term not in seen_terms:
                    seen_add(term)
                    expanded_append(term)

        # Check individual tokens
        tokens = self._tokenize(query_lower)
        for token in tokens:
            token_terms = expansions.get(token)
            if token_terms:
                for term in token_terms:
                    if term not in seen_terms:
                        seen_add(term)
                        expanded_append(term)

        # Check bigrams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            bigram_terms = expansions.get(bigram)
            if bigram_terms:
                for term in bigram_terms:
                    if term not in seen_terms:
                        seen_add(term)
                        expanded_append(term)

        if expanded_terms:
            return query_lower + " " + " ".join(expanded_terms)
        return query_lower

    def add_documents(self, chunk_ids: List[str], texts: List[str]) -> None:
        """Add documents to the BM25 index"""
        for chunk_id, text in zip(chunk_ids, texts):
            self.corpus.append(text)
            self.corpus_ids.append(chunk_id)
            self._tokenized_corpus.append(self._tokenize(text))

    def build_index(self) -> None:
        """Build inverted index with pre-computed IDF and doc lengths."""
        if not self._tokenized_corpus:
            return

        corpus_size = len(self._tokenized_corpus)
        doc_lengths = np.empty(corpus_size, dtype=np.float64)
        nd: Dict[str, int] = {}
        inverted: Dict[str, List[Tuple[int, int]]] = {}

        for doc_idx, tokens in enumerate(self._tokenized_corpus):
            doc_lengths[doc_idx] = len(tokens)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for term, freq in tf.items():
                nd[term] = nd.get(term, 0) + 1
                posting = inverted.get(term)
                if posting is None:
                    inverted[term] = [(doc_idx, freq)]
                else:
                    posting.append((doc_idx, freq))

        avgdl = float(doc_lengths.sum() / corpus_size) if corpus_size > 0 else 0.0

        idf: Dict[str, float] = {}
        idf_sum = 0.0
        negative_idfs: List[str] = []
        for word, freq in nd.items():
            val = math.log(corpus_size - freq + 0.5) - math.log(freq + 0.5)
            idf[word] = val
            idf_sum += val
            if val < 0:
                negative_idfs.append(word)

        average_idf = idf_sum / len(idf) if idf else 0.0
        eps = self._epsilon * average_idf
        for word in negative_idfs:
            idf[word] = eps

        self._inverted_index = inverted
        self._idf = idf
        self._doc_len = doc_lengths
        self._avgdl = avgdl
        self._corpus_size = corpus_size
        self._index_built = True

    def prepare_query(self, query: str) -> str:
        """
        Strip multilingual stopwords from a query before keyword scoring.

        Question words ("how", "como", "warum"), auxiliaries and articles add
        no BM25 signal but do pollute both the score and query expansion — a
        stray "que" can drag in an entire synonym group. Technical identifiers
        and all-caps acronyms are never removed (see ``mcp_server.stopwords``).

        Args:
            query: Raw user query.

        Returns:
            str: Query with stopwords removed, or the original query when
            filtering would leave nothing behind.
        """
        return filter_query_stopwords(query, config.stopword_languages)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search the BM25 index with stopword filtering + query expansion.

        Uses inverted-index posting lists to score only documents containing
        at least one query term. Returns (chunk_id, score) sorted descending.
        """
        if not self._index_built or not self.corpus:
            return []

        # Stopwords are stripped BEFORE expansion so scaffolding words cannot
        # seed synonym lookups.
        filtered_query = self.prepare_query(query)
        expanded_query = self.expand_query(filtered_query)
        tokenized_query = self._tokenize(expanded_query)
        if not tokenized_query:
            return []

        k1 = self._k1
        b = self._b
        avgdl = self._avgdl
        doc_len = self._doc_len
        idf_lookup = self._idf
        inv = self._inverted_index

        candidate_scores: Dict[int, float] = {}
        for q in tokenized_query:
            idf_q = idf_lookup.get(q, 0.0)
            if idf_q == 0.0:
                continue
            posting = inv.get(q)
            if posting is None:
                continue
            for doc_idx, tf in posting:
                dl = doc_len[doc_idx]
                num = tf * (k1 + 1.0)
                den = tf + k1 * (1.0 - b + b * dl / avgdl)
                candidate_scores[doc_idx] = candidate_scores.get(doc_idx, 0.0) + idf_q * (num / den)

        if not candidate_scores:
            return []

        n_candidates = len(candidate_scores)
        if n_candidates <= top_k:
            results = [(self.corpus_ids[idx], score) for idx, score in candidate_scores.items()]
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        indices = np.fromiter(candidate_scores.keys(), dtype=np.intp, count=n_candidates)
        scores = np.fromiter(candidate_scores.values(), dtype=np.float64, count=n_candidates)
        partition_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = partition_idx[np.argsort(scores[partition_idx])[::-1]]
        return [(self.corpus_ids[indices[i]], float(scores[i])) for i in top_indices]

    def clear(self) -> None:
        """Clear the index"""
        self.corpus = []
        self.corpus_ids = []
        self._tokenized_corpus = []
        self._inverted_index = {}
        self._idf = {}
        self._doc_len = None
        self._avgdl = 0.0
        self._corpus_size = 0
        self._index_built = False

    def __len__(self) -> int:
        return len(self.corpus)

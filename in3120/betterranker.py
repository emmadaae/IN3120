#!/usr/bin/python
# -*- coding: utf-8 -*-

from .ranker import Ranker
from .corpus import Corpus
from .posting import Posting
from .invertedindex import InvertedIndex
import math

class BetterRanker(Ranker):
    """
    A ranker that does traditional TF-IDF ranking, possibly combining it with
    a static document score (if present).

    The static document score is assumed accessible in a document field named
    "static_quality_score". If the field is missing or doesn't have a value, a
    default value of 0.0 is assumed for the static document score.

    See Section 7.1.4 in https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf.

    implement a better ranker than the BrainDeadRanker class. For example,
    inverse document frequency (i.e., considering how frequent the query terms
    are across the corpus) and static rank
    (i.e., a query-independent quality score per document) are two factors
    that you should include.
    """

    def __init__(self, corpus: Corpus, inverted_index: InvertedIndex):
        self._score = 0.0
        self._document_id = None
        self._corpus = corpus
        self._inverted_index = inverted_index
        self._dynamic_score_weight = 1.0  # TODO: Make this configurable.
        self._static_score_weight = 1.0  # TODO: Make this configurable.
        # TODO: Make this configurable.
        self._static_score_field_name = "static_quality_score"

    def reset(self, document_id: int) -> None:
        self._score = 0.0
        self._document_id = document_id


    def update(self, term: str, multiplicity: int, posting: Posting) -> None:
        """
        net_rank(q, d) = dynamic_score(q, d) + static_score(d)
        static_score = g(d), for each document d that is query independent,
        quality measure between 0 and 1

        dynamic_score -> query dependent
        """
        assert self._document_id == posting.document_id

        tf = posting.term_frequency

        N = len(self._corpus)
        df = self._inverted_index.get_document_frequency(term)

        self._score += tf * (math.log(N/df))

    def evaluate(self) -> float:

        doc = self._corpus.get_document(self._document_id)
        self._static_score_weight = doc.get_field("static_quality_score", 0.0)

        self._score += self._static_score_weight
        return self._score

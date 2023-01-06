#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools
from .ranker import Ranker
from .corpus import Corpus
from .invertedindex import InvertedIndex
from typing import Iterator, Dict, Any
from .sieve import Sieve
from collections import defaultdict
from collections import Counter


class SimpleSearchEngine:
    """
    A simple implementation of a search core based on an inverted index, suitable for small corpora.
    """

    def __init__(self, corpus: Corpus, inverted_index: InvertedIndex):
        self.__corpus = corpus
        self.__inverted_index = inverted_index


    def evaluate(self, query: str, options: dict, ranker: Ranker) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing N-out-of-M ranked retrieval. I.e., for a supplied query having M
        unique terms, a document is considered to be a match if it contains at least N <= M of those terms.

        The matching documents, if any, are ranked by the supplied ranker, and only the "best" matches are yielded
        back to the client as dictionaries having the keys "score" (float) and "document" (Document).

        The client can supply a dictionary of options that controls this query evaluation process: The value of
        N is inferred from the query via the "match_threshold" (float) option, and the maximum number of documents
        to return to the client is controlled via the "hit_count" (int) option.
        """

        terms_iter_count = Counter([term for term in self.__inverted_index.get_terms(query)])
        terms = [t for t in terms_iter_count.keys()]
        m = len(terms)
        n = max(1, min(m, int(options.get('match_threshold') *m)))
        postings = [self.__inverted_index.get_postings_iterator(i) for i in terms]

        results = []
        dict_postings = {i: next(posting, None) for i, posting in enumerate(postings)}
        dict_postings = {key:value for key,value in dict_postings.items() if value is not None}

        while len(dict_postings) >= n:
            inv_dict = {}
            for key, value in dict_postings.items():
                inv_dict.setdefault(value.document_id, set()).add(key)
            result = list(filter(lambda x: len(x) >= n, inv_dict.values()))

            if len(result) > 0:
                matches = list(result[0])
            else:
                matches = []

            if len(matches) >= n:
                intermediate_result = {key: dict_postings[key] for key in matches}
                if len(results) > 0:
                    if results[-1] != intermediate_result:
                        results.append(intermediate_result)
                else:
                    results.append(intermediate_result)

            doc_id= min(dict_postings, key=lambda x: dict_postings.get(x).document_id)
            dict_postings[doc_id] = next(postings[doc_id], None)
            dict_postings = {k: v for k, v in dict_postings.items() if v is not None}

        results.sort(key=(lambda x: len(x)), reverse=True)

        sieve = Sieve(options.get('hit_count'))
        checked = []

        for i in results:
            current_document = i[list(i.keys())[0]].document_id
            if current_document in checked:
                pass
            else:
                checked.append(current_document)
                ranker.reset(current_document)

                for key, value in i.items():
                    ranker.update(terms[key], terms_iter_count[terms[key]], value)
                sieve.sift(ranker.evaluate(), current_document)

        winners = list(sieve.winners())

        for i in winners:
            yield({'score': i[0], 'document': self.__corpus.get_document(i[1])})

        debug = options.get("debug", False)

        # Document-at-a-time!
        #raise NotImplementedError("You need to implement this as part of the assignment.")

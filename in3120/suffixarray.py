#!/usr/bin/python
# -*- coding: utf-8 -*-

from .corpus import Corpus
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from typing import Any, Dict, Iterator, Iterable, Tuple, List
from .sieve import Sieve
from typing import Iterable, Iterator, List
import itertools
from collections import Counter #impoterte for å bruke i evaluate


class SuffixArray:
    """
    A simple suffix array implementation. Allows us to conduct efficient substring searches.
    The prefix of a suffix is an infix!

    In a serious application we'd make use of least common prefixes (LCPs), pay more attention
    to memory usage, and add more lookup/evaluation features.
    """

    def __init__(self, corpus: Corpus, fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer):
        self.__corpus = corpus
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer
        # The (<document identifier>, <searchable content>) pairs.
        self.__haystack: List[Tuple[int, str]] = []
        # The sorted (<haystack index>, <start offset>) pairs.
        self.__suffixes: List[Tuple[int, int]] = []
        # Constructs __haystack and __suffixes.
        self._counter = Counter() # la til sev for å bruke i evaluate
        self.__build_suffix_array(fields)


    def __build_suffix_array(self, fields: Iterable[str]) -> None:
        """
        Builds a simple suffix array from the set of named fields in the document collection.
        The suffix array allows us to search across all named fields in one go.
        """
        for i in range(self.__corpus.size()):
            doc = self.__corpus.get_document(i)
            text2 = [self.__normalize(doc[field]) for field in fields]
            text = ""
            for t in text2:
                text = text + " $" + t

            self.__haystack.append((doc.document_id, text))
            tokens = []
            for i in self.__tokenizer.tokens(text):
                tokens.append(i)
            tokens.sort()
            for t in tokens:
                self.__suffixes.append((doc.document_id, t[1][0]))

        self.__suffixes.sort(key=(lambda x: self.__haystack[x[0]][1][x[1]:]))


    def __normalize(self, buffer: str) -> str:
        """
        Produces a normalized version of the given string. Both queries and documents need to be
        identically processed for lookups to succeed.
        """
        # Tokenize and join to be robust to nuances in whitespace and punctuation.
        return self.__normalizer.normalize(" ".join(self.__tokenizer.strings(self.__normalizer.canonicalize(buffer))))

    def __binary_search(self, needle: str) -> int:

        """
        Does a binary search for a given normalized query (the needle) in the suffix array (the haystack).
        Returns the position in the suffix array where the normalized query is either found, or, if not found,
        should have been inserted.

        Kind of silly to roll our own binary search instead of using the bisect module, but seems needed
        due to how we represent the suffixes via (index, offset) tuples.
        """

        low = 0
        high = len(self.__suffixes) - 1

        while low < high:
            i = (low + high)//2
            haystack_index = self.__suffixes[i][0]
            text = self.__haystack[haystack_index][1]
            value = text[self.__suffixes[i][1]:self.__suffixes[i][1]+len(needle)]
            if value == needle:
                return i
            elif value < needle:
                low = i+1
            elif value > needle:
                high = i-1
        return low


    def evaluate(self, query: str, options: dict) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing a "phrase prefix search".  E.g., for a supplied query phrase like
        "to the be", we return documents that contain phrases like "to the bearnaise", "to the best",
        "to the behemoth", and so on. I.e., we require that the query phrase starts on a token boundary in the
        document, but it doesn't necessarily have to end on one.

        The matching documents are ranked according to how many times the query substring occurs in the document,
        and only the "best" matches are yielded back to the client. Ties are resolved arbitrarily.

        The client can supply a dictionary of options that controls this query evaluation process: The maximum
        number of documents to return to the client is controlled via the "hit_count" (int) option.

        The results yielded back to the client are dictionaries having the keys "score" (int) and
        "document" (Document).
        """

        if not query:
            return
        phrase = self.__normalize(query)

        n = len(phrase)
        suffix_index = self.__binary_search(phrase)
        haystack_index =self.__suffixes[suffix_index][0]
        haystack_offset = self.__suffixes[suffix_index][1]
        doc_id = self.__haystack[haystack_index][0]
        text = self.__haystack[doc_id][1][haystack_offset: haystack_offset+n]

        if text == phrase:
            self._counter[doc_id] += 1

            current_index = suffix_index -1

            while current_index > 0:
                haystack_index = self.__suffixes[current_index][0]
                haystack_offset = self.__suffixes[current_index][1]
                if self.__haystack[haystack_index][1][haystack_offset: haystack_offset+n] == phrase:
                    self._counter[haystack_index] += 1
                    current_index -= 1
                else:
                    break

            current_index = suffix_index +1

            while current_index < len(self.__suffixes):
                haystack_index = self.__suffixes[current_index][0]
                haystack_offset = self.__suffixes[current_index][1]
                if self.__haystack[haystack_index][1][haystack_offset: haystack_offset+n] == phrase:
                    self._counter[haystack_index] += 1
                    current_index += 1
                else:
                    break

        hit = options.get('hit_count')
        if hit:
            sieve = Sieve(options.get("hit_count"))
        else:
            sieve = Sieve(10)

        for doc_id in self._counter:
            freq = self._counter[doc_id]
            sieve.sift(freq, doc_id)

        for win in sieve.winners():
            doc = self.__corpus.get_document(win[1])
            yield ({'score': win[0], 'document': doc})
        self._counter.clear()

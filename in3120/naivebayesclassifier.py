#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
from .dictionary import InMemoryDictionary
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .corpus import Corpus
from typing import Any, Dict, Iterable, Iterator
from collections import Counter
from typing import Iterable, Iterator, List
import itertools
import math
from collections import defaultdict




class NaiveBayesClassifier:
    """
    Defines a multinomial naive Bayes text classifier.
    """

    def __init__(
        self, training_set: Dict[str, Corpus], fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer
    ):
        """
        Constructor. Trains the classifier from the named fields in the documents in
        the given training set.
        """
        # Used for breaking the text up into discrete classification features.
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer

        # The vocabulary we've seen during training.
        self.__vocabulary = InMemoryDictionary()

        # Maps a category c to the prior probability Pr(c).
        self.__priors: Dict[str, float] = {}

        # Maps a category c and a term t to the conditional probability Pr(t | c).
        self.__conditionals: Dict[str, Dict[str, float]] = {}

        # So that we know how to estimate Pr(t | c) for out-of-vocabulary terms encountered
        # in the texts to classify. Basically the denominators when doing Laplace smoothing,
        # for each category c.
        self.__denominators: Dict[str, int] = {}

        # Train the classifier, i.e., estimate all probabilities.
        self.__compute_priors(training_set)
        self.__compute_vocabulary(training_set, fields)
        self.__compute_posteriors(training_set, fields)

    def __compute_priors(self, training_set):
        """
        Estimates all prior probabilities needed for the naive Bayes classifier.

        maps prior probabilities -- P(c) = Nc/N der Nc: ant doc i class og N er ant doc

        """
        N = 0
        for category in training_set:
            N += len(training_set.get(category))

        for category in training_set:
            Nc = len(training_set.get(category))
            self.__priors[category] = Nc/N



    def __compute_vocabulary(self, training_set, fields):
        """
        Builds up the overall vocabulary as seen in the training set.
        """

        text =""
        for category in training_set:
            text = ""
            for doc in training_set.get(category):
                for field in fields:
                    text += " " + doc[field]
            terms = self.__get_terms(text)
            counter = Counter(terms)

            for term, frequency in counter.items():
                self.__vocabulary.add_if_absent(term)



    def __compute_posteriors(self, training_set, fields):
        """
        Estimates all conditional probabilities needed for the naive Bayes classifier.
        set self.__conditionals and self.__denominators
        where __conditionals -> p(t|c),
        der p(c|t) = antall forekomster av term i dokumenter fra klasse c, /
        summen av term t i vokabulary
        __denominators -> p(t|c) for out-of-vocab terms (laplace smoothing)

        """

        for category in training_set:
            text = ""
            for doc in training_set.get(category):
                for field in fields:
                    text += " " + doc[field]
            terms = self.__get_terms(text)

            counter = Counter(terms)

            denominators = 0
            for term, frequency in counter.items():
                denominators += frequency
            #print("denominators", denominators)
            for term in self.__vocabulary:
                    result = (counter[term[0]] +1 )/(denominators + self.__vocabulary.size())
                    self.__conditionals[(term[0], category)] = result


    def __get_terms(self, buffer):
        """
        Processes the given text buffer and returns the sequence of normalized
        terms as they appear. Both the documents in the training set and the buffers
        we classify need to be identically processed.
        """
        tokens = self.__tokenizer.strings(self.__normalizer.canonicalize(buffer))
        return (self.__normalizer.normalize(t) for t in tokens)


    def classify(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Classifies the given buffer according to the multinomial naive Bayes rule. The computed (score, category) pairs
        are emitted back to the client via the supplied callback sorted according to the scores. The reported scores
        are log-probabilities, to minimize numerical underflow issues. Logarithms are base e.

        The results yielded back to the client are dictionaries having the keys "score" (float) and
        "category" (str).

        1 W ← EXTRACTTOKENSFROMDOC(V, d)
        2 for each c ∈ C
        3 do score[c] ← log prior[c]
        4 for each t ∈ W
        5 do score[c] += log condprob[t][c]
        6 return arg maxc∈ Cscore[c]

        """

        scores = []
        W = []

        terms = self.__get_terms(buffer)

        for t in terms:
            if t in self.__vocabulary:
                W.append(t)

        for category in self.__priors:
            score = math.log(self.__priors[category])
            for term in W:
                score += math.log(self.__conditionals[(term, category)])

            scores.append((score, category))


        scores.sort(key= lambda t: math.fabs(t[0]))

        for score in scores:
            yield({"score": score[0], "category": score[1]})





        #raise NotImplementedError("You need to implement this as part of the assignment.")

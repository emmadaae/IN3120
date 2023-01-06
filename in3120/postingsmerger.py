#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Iterator
from .posting import Posting


class PostingsMerger:
    """
    Utility class for merging posting lists.
    """

    @staticmethod
    def intersection(p1: Iterator[Posting], p2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple AND of two posting lists, given
        iterators over these.

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        list1 = next(p1, None)
        list2 = next(p2, None)

        while list1 and list2:

            if list1.document_id == list2.document_id:
                yield list1
                list1 = next(p1, None)
                list2 = next(p2, None)

            elif list1.document_id < list2.document_id:
                list1 = next(p1, None)

            else:
                list2 = next(p2, None)


    @staticmethod
    def union(p1: Iterator[Posting], p2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple OR of two posting lists, given
        iterators over these.

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        
        """
        list1 = next(p1, None)
        list2 = next(p2, None)

        while list1 and list2:

            if list1.document_id == list2.document_id:
                yield list1
                list1 = next(p1, None)
                list2 = next(p2, None)
            elif list1.document_id < list2.document_id:
                yield list1
                list1 = next(p1, None)
            else:
                yield list2
                list2 = next(p2, None)

        while list1:
            yield list1
            list1 = next(p1, None)

        while list2:
            yield list2
            list2 = next(p2, None)

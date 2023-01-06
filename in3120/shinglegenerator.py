#!/usr/bin/python
# -*- coding: utf-8 -*-

from .tokenizer import Tokenizer
from typing import Iterator, Tuple


class ShingleGenerator(Tokenizer):
    """
    Tokenizes a buffer into overlapping shingles having a specified width. For example, the
    3-shingles for "mouse" are {"mou", "ous", "use"}.
    """

    def __init__(self, width: int):
        assert width > 0
        self.__width = width
        #self.__tokenizer = Tokenizer

    def ranges(self, buffer: str) -> Iterator[Tuple[int, int]]:
        """
        Locates where the shingles begin and end.
        """
        if buffer:
            if self.__width > 0:
                shingles = []
                start = 0
                if self.__width > len(buffer):
                    end = len(buffer)
                    shingles.append((start, end))
                else:
                    end = int(self.__width)
                while end <= len(buffer):
                    if ((start, end)) not in shingles:
                        shingles.append((start, end))
                    start += 1
                    end += 1
                for i in shingles:
                    yield i


        #raise NotImplementedError("You need to implement this as part of the assignment.")

##########################################################################
# Copyright (C) 2022 COAI @ Tsinghua University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#         http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

from collections import Counter
from multiprocessing import Pool
import os
from textwrap import wrap

import torch

from fairseq.data import Dictionary

class TranslationDATDict(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        bos='<s>',
        extra_special_symbols=None,
    ):
        super().__init__(pad=pad, eos=eos, unk=unk, bos=bos, extra_special_symbols=extra_special_symbols)

    @classmethod
    def load(cls, f, seg_tokens=32, extra_symbols=None):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.symbol_start = len(d.symbols)
        d.add_from_file(f)

        # add special plan tokens
        d.first_seg_token = len(d.symbols)
        for i in range(seg_tokens):
            d.add_symbol(f"[P{i}]")
        d.last_seg_token = len(d.symbols)

        if extra_symbols is not None:
            for tok in extra_symbols:
                d.add_symbol(tok)

        return d

    def is_seg_token(self, tok):
        return tok >= self.first_seg_token and tok < self.last_seg_token

    def is_special_token(self, tok):
        return self.is_seg_token(tok) or tok < self.nspecial

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(f, zip(ex_keys + self.symbols[self.symbol_start:self.first_seg_token], 
            ex_vals + self.count[self.symbol_start:self.first_seg_token]))

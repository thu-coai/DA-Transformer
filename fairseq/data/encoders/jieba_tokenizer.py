# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


@dataclass
class JiebaTokenizerConfig(FairseqDataclass):
    pass


@register_tokenizer("jieba", dataclass=JiebaTokenizerConfig)
class JiebaTokenizer(object):
    def __init__(self, cfg: JiebaTokenizerConfig):
        self.cfg = cfg

        try:
            import jieba
            jieba.initialize()

            self.tok = jieba
        except ImportError:
            raise ImportError(
                "Please install Jieba tokenizer with: pip install jieba"
            )

    def encode(self, x: str) -> str:
        return " ".join(self.tok.cut(x))

    def decode(self, x: str) -> str:
        return x.replace(" ","")

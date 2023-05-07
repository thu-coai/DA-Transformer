# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from fairseq.data.encoders import register_tokenizer, build_tokenizer
from fairseq.dataclass import FairseqDataclass


@dataclass
class CompositeTokenizerConfig(FairseqDataclass):
    tokenizer_groups: Optional[Dict[str, Any]] = None


@register_tokenizer("composite", dataclass=CompositeTokenizerConfig)
class CompositeTokenizer(object):
    def __init__(self, cfg: CompositeTokenizerConfig):
        self.cfg = cfg

        self.encoder = build_tokenizer(cfg.tokenizer_groups['encoder'])
        self.decoder = build_tokenizer(cfg.tokenizer_groups['decoder'])

    def encode(self, x: str) -> str:
        return self.encoder.encode(x)

    def decode(self, x: str) -> str:
        return self.decoder.decode(x)


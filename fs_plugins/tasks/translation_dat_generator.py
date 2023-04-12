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

from collections import namedtuple
from json import decoder
import concurrent

import numpy as np
import torch
from fairseq import utils

from collections import namedtuple
DecodeResult = namedtuple("DecodeResult", ['future', 'fn', 'args'])

class TranslationDATGenerator(object):
    def __init__(self, tgt_dict, models=None):
        """
        Generates translations based on directed acyclic graph decoding.

        Args:
            tgt_dict: target dictionary
        """
        self.bos = tgt_dict.bos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.models = models

    @torch.no_grad()
    def generate(self, models, sample, prefix_tokens=None, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the TranslationDATGenerator is not supported"
            )

        for model in models:
            model.eval()
        model = models[0]

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()

        # initialize
        encoder_out = model.forward_encoder([src_tokens, src_lengths])

        prev_decoder_out = model.initialize_output_tokens(encoder_out, src_tokens)
        sent_idxs = torch.arange(bsz)

        prev_decoder_out = prev_decoder_out._replace(
            step=0,
            max_step=0,
        )

        decoder_options = {}
        decoder_out = model.forward_decoder(
            prev_decoder_out, encoder_out, **decoder_options
        )

        if hasattr(decoder_out, "future"): # for overlapped decoding
            return DecodeResult(future=decoder_out.future, fn=decoder_out.fn + [finish_hypo], args=decoder_out.args + [(sent_idxs, self.pad, bsz)])
        else:
            return finish_hypo(decoder_out, sent_idxs, self.pad, bsz)

def finish_hypo(decoder_out, sent_idxs, pad, bsz, step=0):
    finalized = [[] for _ in range(bsz)]

    # collect finalized sentences
    finalized_idxs = sent_idxs
    finalized_tokens = decoder_out.output_tokens
    finalized_scores = decoder_out.output_scores

    def finalized_hypos(step, prev_out_token, prev_out_score):
            cutoff = prev_out_token.ne(pad)
            tokens = prev_out_token[cutoff]
            scores = prev_out_score[cutoff]
            score = scores.mean()

            return {
                "steps": step,
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "hypo_attn": None,
                "alignment": None,
            }

    for i in range(finalized_idxs.size(0)):
        finalized[finalized_idxs[i]] = [
            finalized_hypos(
                step,
                finalized_tokens[i],
                finalized_scores[i],
            )
        ]

    return finalized

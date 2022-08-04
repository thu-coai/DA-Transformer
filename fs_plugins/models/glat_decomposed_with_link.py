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

from fairseq.models.nat.fairseq_nat_model import FairseqNATModel
import logging
import random
import copy
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn, jit
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.modules import (
    PositionalEmbedding,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.nat.nonautoregressive_transformer import NATransformerDecoder
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def torch_seed(seed):
    # modified from lunanlp
    state = torch.random.get_rng_state()
    state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.random.set_rng_state(state_cuda)

# @jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))

@register_model("glat_decomposed_link")
class GlatDecomposedLink(FairseqNATModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.init_beam_search()

    def init_beam_search(self):
        if self.args.decode_strategy == "beamsearch":
            import dag_search
            self.dag_search = dag_search
            dag_search.beam_search_init(self.args.decode_max_batchsize, self.args.decode_beamsize,
                    self.args.decode_top_cand_n, self.decoder.max_positions(), self.tgt_dict, self.args.decode_lm_path)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = GlatLinkDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        GlatLinkDecoder.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )

        parser.add_argument('--links-feature', type=str, default="feature:position", help="Features used to predict transition.")
        parser.add_argument('--max-transition-length', type=int, default=99999, help="Max transition distance. -1 means no limitation, \
                        which cannot be used for cuda custom operations. To use cuda operations with no limitation, please use a very large number such as 99999.")

        parser.add_argument("--src-upsample-scale", type=float, default=None, help="Specify the graph size with a upsample factor (lambda).  Graph Size = \\lambda * src_length")
        parser.add_argument("--src-upsample-fixed", type=int, default=None, help="Specify the graph size by a constant. Cannot use together with src-upsample-scale")
        parser.add_argument("--length-multiplier", type=float, default=None, help="Deprecated") # does not work now
        parser.add_argument('--max-decoder-batch-tokens', type=int, default=None, help="Max tokens for LightSeq Decoder when using --src-upsample-fixed")

        parser.add_argument('--filter-max-length', default=None, type=str, help='Filter the sample that above the max lengths, e.g., "128:256" indicating 128 for source, 256 for target. Default: None, for filtering according max-source-positions and max-target-positions')
        parser.add_argument("--filter-ratio", type=float, default=None, help="Deprecated") # does not work now; need support of trainer.py

        parser.add_argument('--decode-strategy', type=str, default="lookahead", help='One of "greedy", "lookahead", "beamsearch"')

        parser.add_argument('--decode-alpha', type=float, default=1.1, help="Used for length penalty. Beam Search finds the sentence maximize: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]")
        parser.add_argument('--decode-beta', type=float, default=1, help="Scale the score of logits. log P(Y, A) := sum P(y_i|a_i) + beta * sum log(a_i|a_{i-1})")
        parser.add_argument('--decode-top-cand-n', type=float, default=5, help="Numbers of top candidates when considering transition")
        parser.add_argument('--decode-gamma', type=float, default=0.1, help="Used for n-gram language model score. Beam Search finds the sentence maximize: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]")
        parser.add_argument('--decode-beamsize', type=float, default=100, help="Beam size")
        parser.add_argument('--decode-max-beam-per-length', type=float, default=10, help="Limits the number of beam that has a same length in each step")
        parser.add_argument('--decode-top-p', type=float, default=0.9, help="Max probability of top candidates when considering transition")
        parser.add_argument('--decode-lm-path', type=str, default=None, help="Path to n-gram language model. None for not using n-gram LM")
        parser.add_argument('--decode-max-batchsize', type=int, default=32, help="Should not be smaller than the real batch size (the value is used for memory allocation)")
        parser.add_argument('--decode-dedup', type=bool, default=False, help="Use token deduplication in BeamSearch")

    def extract_valid_links(self, content, valid_mask):
        # batch * prelen * prelen * chunk, batch * prelen

        prelen = content.shape[1]
        translen: int = self.args.max_transition_length
        if translen > prelen - 1:
            translen = prelen - 1
        valid_links_idx = torch.arange(prelen, dtype=torch.long, device=content.device).unsqueeze(1) + \
                    torch.arange(translen, dtype=torch.long, device=content.device).unsqueeze(0) + 1
        invalid_idx_mask = valid_links_idx >= valid_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
        valid_links_idx = valid_links_idx.unsqueeze(0).masked_fill(invalid_idx_mask, 0)

        res = content.gather(2, valid_links_idx.unsqueeze(-1).expand(-1, -1, -1, content.shape[-1]))
        res.masked_fill_(invalid_idx_mask.unsqueeze(-1), float("-inf"))

        return res, invalid_idx_mask.all(-1) # batch * prelen * trans_len * chunk, batch * prelen * trans_len

    def restore_valid_links(self, links):
        # batch * prelen * trans_len
        batch_size, prelen, translen = links.shape
        translen: int = self.args.max_transition_length
        if translen > prelen - 1:
            translen = prelen - 1
        valid_links_idx = torch.arange(prelen, dtype=torch.long, device=links.device).unsqueeze(1) + \
                    torch.arange(translen, dtype=torch.long, device=links.device).unsqueeze(0) + 1
        invalid_idx_mask = valid_links_idx >= prelen
        valid_links_idx.masked_fill_(invalid_idx_mask, prelen)
        res = torch.zeros(batch_size, prelen, prelen + 1, dtype=torch.float, device=links.device).fill_(float("-inf"))
        res.scatter_(2, valid_links_idx.unsqueeze(0).expand(batch_size, -1, -1), links)
        return res[:, :, :prelen]

    def extract_links(self, features, prev_output_tokens,
            link_positional, query_linear, key_linear, gate_linear):

        links_feature = vars(self.args).get("links_feature", "feature:position").split(":")

        links_feature_arr = []
        if "feature" in links_feature:
            links_feature_arr.append(features)
        if "position" in links_feature or "sinposition" in links_feature:
            links_feature_arr.append(link_positional(prev_output_tokens))

        features_withpos = torch.cat(links_feature_arr, dim=-1)

        batch_size = features.shape[0]
        seqlen = features.shape[1]
        chunk_num = self.args.decoder_attention_heads
        chunk_size = self.args.decoder_embed_dim // self.args.decoder_attention_heads
        ninf = float("-inf")
        target_dtype = torch.float

        query_chunks = query_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        key_chunks = key_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        log_gates = F.log_softmax(gate_linear(features_withpos), dim=-1, dtype=target_dtype) # batch_size * seqlen * chunk_num
        log_multi_content = (torch.einsum("bicf,bjcf->bijc", query_chunks.to(dtype=target_dtype), key_chunks.to(dtype=target_dtype)) / (chunk_size ** 0.5))

        if self.args.max_transition_length != -1:
            log_multi_content_extract, link_nouse_mask = self.extract_valid_links(log_multi_content, prev_output_tokens.ne(self.pad))
                    # batch * seqlen * trans_len * chunk_num, batch * seqlen * trans_len
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            log_multi_content_extract = F.log_softmax(log_multi_content_extract, dim=2)
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            links = logsumexp(log_multi_content_extract + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * trans_len
        else:
            link_mask = torch.ones(seqlen, seqlen, device=prev_output_tokens.device, dtype=bool).triu_(1).unsqueeze(0) & prev_output_tokens.ne(self.pad).unsqueeze(1)
            link_nouse_mask = link_mask.sum(dim=2, keepdim=True) == 0
            link_mask.masked_fill_(link_nouse_mask, True)
            log_multi_content.masked_fill_(~link_mask.unsqueeze(-1), ninf)
            log_multi_attention = F.log_softmax(log_multi_content, dim=2)
            log_multi_attention = log_multi_attention.masked_fill(link_nouse_mask.unsqueeze(-1), ninf)
            links = logsumexp(log_multi_attention + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * seqlen

        return links

    def extract_features(self, prev_output_tokens, encoder_out, rand_seed, require_links=False):
        with torch_seed(rand_seed):
            features, _ = self.decoder.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                embedding_copy=False
            )
            # word_ins_out = self.decoder.output_layer(features)
            word_ins_out = self.decoder.output_projection(features)

            links = None
            if require_links:
                links = self.extract_links(features, \
                            prev_output_tokens, \
                            self.decoder.link_positional, \
                            self.decoder.query_linear, \
                            self.decoder.key_linear, \
                            self.decoder.gate_linear
                        )

        return word_ins_out, links

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat=None, glat_function=None, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        rand_seed = random.randint(0, 19260817)
        # decoding
        glat_info = None
        if glat and tgt_tokens is not None:
            with torch.set_grad_enabled(glat.get('require_glance_grad', False)):
                word_ins_out, links = self.extract_features(prev_output_tokens, encoder_out, rand_seed, require_links=True)
                prev_output_tokens, tgt_tokens, glat_info = glat_function(self, word_ins_out, tgt_tokens, prev_output_tokens, glat, links=links)
                word_ins_out = None

        word_ins_out, links = self.extract_features(prev_output_tokens, encoder_out, rand_seed, require_links=True)

        ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "nll_loss": True,
            }
        }
        ret['links'] = links

        ret["length"] = {
            "out": length_out,
            "tgt": length_tgt,
            "factor": self.decoder.length_loss_factor,
        }
        if glat_info is not None:
            ret.update(glat_info)
        return ret


    def initialize_output_tokens_with_length(self, src_tokens, length_tgt):
        max_length = length_tgt.max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        return initial_output_tokens

    def initialize_output_tokens_upsample_by_tokens(self, src_tokens):
        if vars(self.args).get("src_upsample_scale", None) is not None:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            length_tgt = (length_tgt * self.args.src_upsample_scale).long().clamp_(min=2)
        else:
            length_tgt = torch.zeros(src_tokens.shape[0], device=src_tokens.device, dtype=src_tokens.dtype).fill_(self.args.src_upsample_fixed)
        return self.initialize_output_tokens_with_length(src_tokens, length_tgt)

    def initialize_output_tokens_multiplier_by_tokens(self, src_tokens, tgt_tokens):
        length_tgt = torch.sum(tgt_tokens.ne(self.tgt_dict.pad_index), -1)
        length_tgt = (length_tgt * self.args.length_multiplier).long().clamp_(min=2)
        return self.initialize_output_tokens_with_length(src_tokens, length_tgt)

    def initialize_output_tokens_by_tokens(self, src_tokens, tgt_tokens):
        if vars(self.args).get("src_upsample_scale", None) is not None or vars(self.args).get("src_upsample_fixed", None) is not None:
            return self.initialize_output_tokens_upsample_by_tokens(src_tokens)
        elif vars(self.args).get("length_multiplier", None) is not None:
            return self.initialize_output_tokens_multiplier_by_tokens(src_tokens, tgt_tokens)

    def initialize_output_tokens_upsample(self, encoder_out, src_tokens):
        # length prediction
        if vars(self.args).get("src_upsample_scale", None) is not None:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            length_tgt = (length_tgt * self.args.src_upsample_scale).long().clamp_(min=2)
        else:
            length_tgt = torch.zeros(src_tokens.shape[0], device=src_tokens.device, dtype=src_tokens.dtype).fill_(self.args.src_upsample_fixed)
        initial_output_tokens = self.initialize_output_tokens_with_length(src_tokens, length_tgt)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def initialize_output_tokens_multiplier(self, encoder_out, src_tokens):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )
        length_tgt = (length_tgt * self.args.length_multiplier).long().clamp_(min=2)
        initial_output_tokens = self.initialize_output_tokens_with_length(src_tokens, length_tgt)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        if vars(self.args).get("src_upsample_scale", None) is not None or vars(self.args).get("src_upsample_fixed", None) is not None:
            return self.initialize_output_tokens_upsample(encoder_out, src_tokens)
        elif vars(self.args).get("length_multiplier", None) is not None:
            return self.initialize_output_tokens_multiplier(encoder_out, src_tokens)

    def max_positions(self):
        if vars(self.args).get("filter_max_length", None) is not None:
            if ":" not in self.args.filter_max_length:
                a = b = int(self.args.filter_max_length)
            else:
                a, b = self.args.filter_max_length.split(":")
                a, b = int(a), int(b)
            return (a, b)
        else:
            if vars(self.args).get("src_upsample_fixed", None) is not None:
                return (self.encoder.max_positions(), self.decoder.max_positions())
            elif vars(self.args).get("src_upsample_scale", None) is not None:
                return (min(self.encoder.max_positions(), int(self.decoder.max_positions() / self.args.src_upsample_scale)), self.decoder.max_positions())
            else:
                return (min(self.encoder.max_positions(), int(self.decoder.max_positions() / self.args.length_multiplier)), self.decoder.max_positions())

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens

        history = decoder_out.history
        rand_seed = random.randint(0, 19260817)

        # execute the decoder
        output_logits, links = self.extract_features(output_tokens, encoder_out, rand_seed, require_links=True)
        if self.args.max_transition_length != -1:
            links = self.restore_valid_links(links)
        output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)

        output_logits_normalized = output_logits.log_softmax(dim=-1)
        unreduced_logits, unreduced_tokens = output_logits_normalized.max(dim=-1)
        unreduced_tokens = unreduced_tokens.tolist()

        if self.args.decode_strategy in ["lookahead", "greedy"]:
            if self.args.decode_strategy == "lookahead":
                output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                links_idx = (links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[1].cpu().tolist() # batch * prelen
            elif self.args.decode_strategy == "greedy":
                output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                links_idx = links.max(dim=-1)[1].cpu().tolist() # batch * prelen

            unpad_output_tokens = []
            for i, length in enumerate(output_length):
                last = unreduced_tokens[i][0]
                j = 0
                res = [last]
                while j != length - 1:
                    j = links_idx[i][j]
                    now_token = unreduced_tokens[i][j]
                    if now_token != self.tgt_dict.pad_index and now_token != last:
                        res.append(now_token)
                    last = now_token
                unpad_output_tokens.append(res)

            output_seqlen = max([len(res) for res in unpad_output_tokens])
            output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
            output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
        elif self.args.decode_strategy == "beamsearch":

            batch_size, prelen, _ = links.shape

            assert batch_size <= self.args.decode_max_batchsize, "Please set --decode-max-batchsize for beamsearch with a larger batch size"

            top_logits, top_logits_idx = output_logits.log_softmax(dim=-1).topk(self.args.decode_top_cand_n, dim=-1)
            dagscores_arr = (links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
            dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n

            nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
            logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
            idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
            logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx] # batch * prelen * top_cand_n

            rearange_idx = logits_idx.sort(dim=-1)[1]
            dagscores = dagscores.gather(-1, rearange_idx) # batch * prelen * top_cand_n
            nextstep_idx = nextstep_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
            logits_idx = logits_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n

            dagscores = np.ascontiguousarray(dagscores.cpu().numpy())
            nextstep_idx = np.ascontiguousarray(nextstep_idx.int().cpu().numpy())
            logits_idx = np.ascontiguousarray(logits_idx.int().cpu().numpy())
            output_length_cpu = np.ascontiguousarray(output_length.int().cpu().numpy())

            res, score = self.dag_search.dag_search(dagscores, nextstep_idx, logits_idx,
                output_length_cpu,
                self.args.decode_alpha,
                self.args.decode_gamma,
                self.args.decode_beamsize,
                self.args.decode_max_beam_per_length,
                self.args.decode_top_p,
                self.tgt_dict.pad_index,
                self.tgt_dict.bos_index,
                1 if self.args.decode_dedup else 0
            )
            output_tokens = torch.tensor(res, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
            output_scores = torch.tensor(score, device=decoder_out.output_scores.device, dtype=decoder_out.output_scores.dtype).unsqueeze(dim=-1).expand(*output_tokens.shape)

        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=torch.full(output_tokens.size(), 1.0),
            attn=None,
            history=history,
        )


class GlatLinkDecoder(NATransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.init_link_feature(args)

    def init_link_feature(self, args):
        links_feature = self.args.links_feature.split(":")
        links_dim = 0
        if "feature" in links_feature:
            links_dim += args.decoder_embed_dim
        if "position" in links_feature:
            self.link_positional = PositionalEmbedding(args.max_target_positions, args.decoder_embed_dim, self.padding_idx, True)
            links_dim += args.decoder_embed_dim
        elif "sinposition" in links_feature:
            self.link_positional = PositionalEmbedding(args.max_target_positions, args.decoder_embed_dim, self.padding_idx, False)
            links_dim += args.decoder_embed_dim
        else:
            self.link_positional = None

        self.query_linear = nn.Linear(links_dim, args.decoder_embed_dim)
        self.key_linear = nn.Linear(links_dim, args.decoder_embed_dim)
        self.gate_linear = nn.Linear(links_dim, args.decoder_attention_heads)

    @staticmethod
    def add_args(parser):
        pass

@register_model_architecture(
    "glat_decomposed_link", "glat_decomposed_link_6e6d512"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)

@register_model_architecture(
    "glat_decomposed_link", "glat_decomposed_link_base"
)
def base_architecture2(args):
    base_architecture(args)

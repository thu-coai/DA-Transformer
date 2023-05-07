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
import concurrent
import time
import json
import os
import argparse
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        state_cuda = torch.cuda.random.get_rng_state()
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state(state_cuda)

# @jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))

# Due to the use of multi-processing, beamsearch functions are in global scope
def init_beam_search(*args):
    import dag_search
    dag_search.beam_search_init(*args)

def call_dag_search(*args):
    import dag_search
    res, score = dag_search.dag_search(*args)
    output_tokens = torch.tensor(res)
    output_scores = torch.tensor(score).unsqueeze(-1).expand_as(output_tokens)
    return output_tokens, output_scores

def subprocess_init(n):
    time.sleep(10) # Do something to wait all subprocess to start
    print(f"overlapped decoding: subprocess {n} initializing", flush=True)
    return n

from collections import namedtuple
DecodeResult = namedtuple("DecodeResult", ['future', 'fn', 'args'])

@register_model("glat_decomposed_link")
class GlatDecomposedLink(FairseqNATModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.src_dict = encoder.dictionary
        self.init_beam_search()

    def init_beam_search(self):
        if self.args.decode_strategy == "beamsearch":
            if self.args.decode_max_workers >= 1: # overlapped decoding
                import multiprocessing as mp
                ctx = mp.get_context('spawn')
                self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.args.decode_max_workers, mp_context=ctx, initializer=init_beam_search,
                    initargs=(self.args.decode_max_batchsize, self.args.decode_beamsize, self.args.decode_top_cand_n,
                              self.decoder.max_positions(), self.args.max_decoder_batch_tokens, self.args.decode_threads_per_worker,
                              self.tgt_dict, self.args.decode_lm_path))
                for x in self.executor.map(subprocess_init, range(self.args.decode_max_workers)):
                    pass
            else: # vanilla decoding
                init_beam_search(self.args.decode_max_batchsize, self.args.decode_beamsize, self.args.decode_top_cand_n,
                                 self.decoder.max_positions(), self.args.max_decoder_batch_tokens, self.args.decode_threads_per_worker,
                                 self.tgt_dict, self.args.decode_lm_path)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        **kwargs,
    ):
        from .hub_interface import DATHubInterface

        from fairseq import checkpoint_utils, file_utils

        model_path = file_utils.load_archive_file(model_name_or_path)

        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            config = json.load(open(config_path, 'r'))
            for key, value in config.items():
                if key not in kwargs:
                    kwargs[key] = value
        kwargs["data"] = model_path

        # convenience hack for loading data and BPE codes from model archive
        # if data_name_or_path.startswith("."):
        #     kwargs["data"] = os.path.abspath(os.path.join(model_path, data_name_or_path))
        # else:
        #     kwargs["data"] = file_utils.load_archive_file(data_name_or_path)
        for file, arg in {
            "code": "bpe_codes",
            "bpecodes": "bpe_codes",
            "sentencepiece.bpe.model": "sentencepiece_model",
            "merges.txt": "bpe_merges",
            "vocab.json": "bpe_vocab",
        }.items():
            path = os.path.join(model_path, file)
            if os.path.exists(path):
                kwargs[arg] = path

        # utils.import_user_module(argparse.Namespace(user_dir=f"{os.path.dirname(os.path.abspath(__file__))}/../"))

        models, args, task = checkpoint_utils.load_model_ensemble_and_task(
            [os.path.join(model_path, kwargs['checkpoint_file'])],
            arg_overrides=kwargs,
        )
        return DATHubInterface(args, task, models[0])

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = GlatLinkDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)

        if args.load_pretrained_model is not None:
            logging.info(f"Loading pretrained model from {args.load_pretrained_model}")
            states = torch.load(args.load_pretrained_model, map_location='cpu')
            extracted = True
            if 'model' in states and 'args' in states:
                states = states['model']
                extracted = False

            ## expanding embed_position
            for position_name, target_position_length in [
                ('encoder.embed_positions.weight', model.encoder.embed_positions.weight.size(0)), \
                ('decoder.embed_positions.weight', model.decoder.embed_positions.weight.size(0))]:
                if states[position_name].size(0) < target_position_length:
                    logging.warn("Current max position length is longer than that of pre-trained checkpoints. Automatically extended...")
                    _index = torch.arange(states[position_name].size(1))
                    expand_position_states = states[position_name].clone()
                    while states[position_name].size(0) < target_position_length:
                        _index = torch.cat((_index[1:], _index[:1]), dim=0)
                        states[position_name] = torch.cat([states[position_name], expand_position_states[:, _index]],
                                                          dim=0)
                if states[position_name].size(0) > target_position_length:
                    logging.warn("Current max position length is shorter than that of pre-trained checkpoints. Automatically truncated...")
                    states[position_name] = states[position_name][:target_position_length]

            ## modifying parameter names for compatibility
            for name in list(states.keys()):
                if name.startswith("decoder.link_predictor."):
                    states[name.replace("decoder.link_predictor.", "decoder.")] = states[name]
                    states.pop(name)

            if not args.segment_embedding:
                if "decoder.embed_seg.weight" in states:
                    logging.warn("--segment-embedding disabled. dropping decoder.embed_seg.weight ...")
                    states.pop("decoder.embed_seg.weight")
            else:
                ckpt_seg = states["decoder.embed_seg.weight"]
                init_seg = model.decoder.embed_seg.weight.data
                copy_num = min(init_seg.shape[0], ckpt_seg.shape[0])
                init_seg[:copy_num].copy_(ckpt_seg[:copy_num])
                states["decoder.embed_seg.weight"] = init_seg

            # Compatible (debug)
            # if not extracted:
            #     ckpt_pos = states["decoder.embed_positions.weight"]
            #     init_pos = model.decoder.embed_positions.weight.data.zero_()
            #     init_pos[2:].copy_(ckpt_pos[1:-1])
            #     states["decoder.embed_positions.weight"] = init_pos

            if "encoder.layers.0.para" in states.keys():
                logging.warn("Automatically converting lightseq transformer to fairseq transformer. "
                    "Please make sure that the architecture is matched; otherwise it may cause unexpected behaviours.")
                from ..scripts._convert_utils import convert_state_from_ls_to_fs
                states = convert_state_from_ls_to_fs(states, args.decoder_embed_dim, args.decoder_ffn_embed_dim, args.decoder_layers)

            model.load_state_dict(states)
            args.load_pretrained_model = None  # Clear this param

        return model

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
            help='Weights on the length prediction loss. Required if --upsample_base "predict" is set.',
        )

        parser.add_argument('--load-pretrained-model', type=str, default=None, help='Path to a file containing a pre-trained model.')

        parser.add_argument('--links-feature', type=str, default="feature:position", help='Specifies the features used to predict transitions, separated by a colon. '
                         'For example, "feature:position" represents the concatenation of decoder features and learnable positional embeddings.')
        parser.add_argument('--segment-embedding', action='store_true', default=False,
                    help='Adds an additional embedding represented segment id for the decoder input.')
        parser.add_argument('--max-transition-length', type=int, default=99999,
                    help='Specifies the maximum transition distance. A value of -1 indicates no limit, but this cannot be used with CUDA custom operations. '
                        'To use CUDA operations with no limit, specify a very large number such as 99999.')
        parser.add_argument('--filter-max-length', default=None, type=str,
                    help='Filters samples that exceed the maximum lengths. For example, "128:256" indicates a maximum source length of 128 and a maximum target length of 256. '
                        'The default value of None filters according to max-source-positions and max-target-positions.')

        try:
            # shared arguments with task, can be altered after training
            parser.add_argument('--max-encoder-batch-tokens', type=int, default=None,
                    help='Specifies the maximum number of tokens for the encoder input to avoid running out of memory. The default value of None indicates no limit.')
            parser.add_argument('--max-decoder-batch-tokens', type=int, default=None,
                    help='Specifies the maximum number of tokens for the decoder input to avoid running out of memory. The default value of None indicates no limit.')

            parser.add_argument("--upsample-base", type=str, default="source", help='Possible values are: ["predict", "source", "source_old"]. '
                'If set to "predict", the DAG size will be determined by the golden target length during training and the predicted length during inference. Note that --length-loss-factor must be greater than 0 during training. '
                'If set to "source", the DAG size will be determined by the source length during both training and inference. You can disable the length predictor during training by setting --length-loss-factor to 0. '
                'If set to "source_old", the DAG size length is determined similarly to "source" but several token longer. This option is only used for compatibility with the upsampling method in version 1.0.')
            parser.add_argument("--decode-upsample-scale", type=float, default=None, help="Up-sampling scale to determine the DAG size during inference. "
                        "If --upsample-scale used in training is a fixed number, this parameter should be the same value."
                        "If --upsample-scale used in training is a range, this parameter can be the average of the range, or tuned on the validation set.")
            parser.add_argument('--decode-strategy', type=str, default="lookahead",
                        help='Decoding strategy to use. Options include "greedy", "lookahead", "viterbi", "jointviterbi", "sample", and "beamsearch".')

            parser.add_argument('--decode-no-consecutive-repeated-ngram', type=int, default=0,
                        help="Prevent consecutive repeated k-grams (k <= n) in the generated text. Use 0 to disable this feature. This argument is used in greedy, lookahead, sample, and beam search decoding methods.")
            parser.add_argument('--decode-no-repeated-ngram', type=int, default=0,
                        help="Prevent repeated k-grams (not necessarily consecutive) with order n or higher in the generated text. Use 0 to disable this feature. "
                                "This argument is used in lookahead, sample, and beam search decoding methods.")
            parser.add_argument('--decode-top-cand-n', type=float, default=5,
                        help='Number of top candidates to consider during transition. This argument is used in lookahead decoding with n-gram prevention, and sample and beamsearch decoding methods.')
            parser.add_argument('--decode-top-p', type=float, default=0.9,
                        help="Maximum probability of top candidates to consider during transition. This argument is used in lookahead decoding with n-gram prevention, and sample and beamsearch decoding methods.")
            parser.add_argument('--decode-viterbibeta', type=float, default=1,
                        help="Parameter used for length penalty in Viterbi decoding. The sentence with the highest score is found using: P(A,Y|X) / |Y|^{beta}")
            parser.add_argument('--decode-temperature', type=float, default=1, help="Temperature to use in sample decoding.")

            parser.add_argument('--decode-beamsize', type=float, default=100, help="Beam size used in beamsearch decoding.")
            parser.add_argument('--decode-max-beam-per-length', type=float, default=10,
                        help="Maximum number of beams with the same length in each step during beamsearch decoding.")
            parser.add_argument('--decode-gamma', type=float, default=0.1,
                                help="Parameter used for n-gram language model score in beamsearch decoding. The sentence with the highest score "
                                    "is found using: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]")
            parser.add_argument('--decode-alpha', type=float, default=1.1,
                                help="Parameter used for length penalty in beamsearch decoding. "
                                    "The sentence with the highest score is found using: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]")
            parser.add_argument('--decode-beta', type=float, default=1, help="Parameter used to scale the score of logits in beamsearch decoding. "
                                    "The score of a sentence is given by: sum P(y_i|a_i) + beta * sum log(a_i|a_{i-1})")
            parser.add_argument('--decode-lm-path', type=str, default=None, help="Path to n-gram language model to use during beamsearch decoding. Set to None to disable n-gram LM.")
            parser.add_argument('--decode-max-batchsize', type=int, default=32, help="Maximum batch size to use during beamsearch decoding. "
                                    "Should not be smaller than the actual batch size, as it is used for memory allocation.")
            parser.add_argument('--decode-max-workers', type=int, default=0, help="Number of multiprocess workers to use during beamsearch decoding. "
                                    'More workers will consume more memory. It does not affect decoding latency but decoding throughtput, '
                                    'so you must use "fariseq-fastgenerate" to enable the overlapped decoding to tell the difference.')
            parser.add_argument('--decode-threads-per-worker', type=int, default=4, help="Number of threads per worker to use during beamsearch decoding. "
                                    "This setting also applies to both vanilla decoding and overlapped decoding. A value between 2 and 8 is typically optimal.")
            parser.add_argument('--decode-dedup', type=bool, default=False, help="Enable token deduplication in BeamSearch.")
        except:
            pass

    def extract_valid_links(self, content, valid_transition_mask):
        # content: batch * prelen * prelen * chunk
        # valid_transition_mask: batch * prelen * prelen

        prelen = content.shape[1]
        translen: int = self.args.max_transition_length
        if translen > prelen - 1:
            translen = prelen - 1
        valid_links_idx = torch.arange(prelen, dtype=torch.long, device=content.device).unsqueeze(1) + \
                    torch.arange(translen, dtype=torch.long, device=content.device).unsqueeze(0) + 1
        valid_links_idx = valid_links_idx.unsqueeze(0)

        res = content.masked_fill_(~valid_transition_mask.unsqueeze(-1), float("-inf")).\
            gather(2, valid_links_idx.clip(max=prelen-1).unsqueeze(-1).expand(content.shape[0], -1, -1, content.shape[-1])).\
            masked_fill_((valid_links_idx >= prelen).unsqueeze(-1), float("-inf"))

        return res, ~(valid_transition_mask.any(-1))

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
        res = torch.zeros(batch_size, prelen, prelen + 1, dtype=links.dtype, device=links.device).fill_(float("-inf"))
        res.scatter_(2, valid_links_idx.unsqueeze(0).expand(batch_size, -1, -1), links)
        return res[:, :, :prelen]

    def extract_links(self, features, prev_output_tokens,
            link_positional, query_linear, key_linear, gate_linear, net_input=None, training=True):
        # feature: [batch_size, prelen, hidden_size]
        # prev_output_tokens: [batch_size, prelen]

        links_feature = vars(self.args).get("links_feature", "feature:position").split(":")

        links_feature_arr = []
        if "feature" in links_feature:
            links_feature_arr.append(features) # add transformer decoder feature
        if "position" in links_feature or "sinposition" in links_feature:
            links_feature_arr.append(link_positional(prev_output_tokens)) # add learnable / absolute positional embeddings

        features_withpos = torch.cat(links_feature_arr, dim=-1)

        batch_size = features.shape[0]
        prelen = features.shape[1]
        chunk_num = self.args.decoder_attention_heads
        chunk_size = self.args.decoder_embed_dim // self.args.decoder_attention_heads
        ninf = float("-inf")

        if training: # use higher precision in training
            target_dtype = torch.float
            logsumexp_fast = logsumexp
        else:
            target_dtype = query_linear.weight.dtype
            logsumexp_fast = torch.logsumexp

        # Use multiple heads in calculating transition matrix
        query_chunks = query_linear(features_withpos).reshape(batch_size, prelen, chunk_num, chunk_size)
        key_chunks = key_linear(features_withpos).reshape(batch_size, prelen, chunk_num, chunk_size)
        # The head probability on each position. log_gates: batch_size * prelen * chunk_num
        log_gates = F.log_softmax(gate_linear(features_withpos), dim=-1, dtype=target_dtype)
        # Transitition probability for each head. log_multi_content: batch_size * prelen * prelen * chunk_num
        log_multi_content = (torch.einsum("bicf,bjcf->bijc", query_chunks.to(dtype=target_dtype), key_chunks.to(dtype=target_dtype)) / (chunk_size ** 0.5))

        # transition_valid_mask specifies all possible transition places for each position
        # transition_valid_mask shape: [batch_size, prelen, prelen]
        if net_input is not None and "bound_end" in net_input and net_input['bound_end'] is not None: # if bound_end is prepared in net_input
            bound_end = net_input["bound_end"] # batch_size * prelen
            valid_links_idx = torch.arange(prelen, dtype=torch.long, device=features.device).unsqueeze(0).unsqueeze(0)
            transition_valid_mask = valid_links_idx <= bound_end.unsqueeze(-1)
        else: # inferred from prev_output_tokens
            transition_valid_mask = prev_output_tokens.ne(self.pad).unsqueeze(1)
        # only allows left-to-right transition
        transition_valid_mask = transition_valid_mask & torch.ones(prelen, prelen, device=prev_output_tokens.device, dtype=bool).triu_(1).unsqueeze(0)

        if self.args.max_transition_length != -1: # finity transtion length, prepare for cuda input format
            log_multi_content_extract, link_nouse_mask = self.extract_valid_links(log_multi_content, transition_valid_mask)
                    # batch * prelen * trans_len * chunk_num, batch * prelen * trans_len
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            log_multi_content_extract = F.log_softmax(log_multi_content_extract, dim=2)
            log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
            links = logsumexp_fast(log_multi_content_extract + log_gates.unsqueeze(2), dim=-1) # batch_size * prelen * trans_len
        else: # infinility transition length, prepare for torch input format
            link_nouse_mask = transition_valid_mask.sum(dim=2, keepdim=True) == 0
            transition_valid_mask.masked_fill_(link_nouse_mask, True)
            log_multi_content.masked_fill_(~transition_valid_mask.unsqueeze(-1), ninf)
            log_multi_attention = F.log_softmax(log_multi_content, dim=2)
            log_multi_attention = log_multi_attention.masked_fill(link_nouse_mask.unsqueeze(-1), ninf)
            links = logsumexp_fast(log_multi_attention + log_gates.unsqueeze(2), dim=-1) # batch_size * prelen * prelen

        return links

    def extract_features(self, prev_output_tokens, encoder_out, net_input, rand_seed, require_links=False, training=True):
        with torch_seed(rand_seed):
            features, _ = self.decoder.extract_features(
                prev_output_tokens,
                net_input,
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
                            self.decoder.gate_linear,
                            net_input=net_input,
                            training=training
                        )

        return word_ins_out, links

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, net_input=None, glat=None, glat_function=None, **kwargs
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

        rand_seed = np.random.randint(0, 19260817)
        # decoding
        glat_info = None
        if glat and tgt_tokens is not None:
            with torch.set_grad_enabled(glat.get('require_glance_grad', False)):
                word_ins_out, links = self.extract_features(prev_output_tokens, encoder_out, net_input, rand_seed, require_links=True)
                prev_output_tokens, tgt_tokens, glat_info = glat_function(self, word_ins_out, tgt_tokens, prev_output_tokens, net_input, glat, links=links)
                word_ins_out = None

        word_ins_out, links = self.extract_features(prev_output_tokens, encoder_out, net_input, rand_seed, require_links=True)

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

    def initialize_output_tokens(self, encoder_out, src_tokens, max_clip=None):
        max_clip = self.decoder.max_positions() - 1
        if self.args.max_decoder_batch_tokens is not None:
            max_clip = min(max_clip, self.args.max_decoder_batch_tokens // src_tokens.shape[0])

        if self.args.upsample_base == "source":
            length_src = torch.sum(src_tokens.ne(self.src_dict.pad_index), -1)
            length_src_special = torch.sum(src_tokens.eq(self.src_dict.bos_index) | src_tokens.eq(self.src_dict.eos_index), -1)
            length_tgt = ((length_src - length_src_special) * self.args.decode_upsample_scale).long().clip_(min=0) + length_src_special
        elif self.args.upsample_base == "source_old":
            # for compatibility with old checkpoints (counting special tokens in upsampling)
            length_src = torch.sum(src_tokens.ne(self.src_dict.pad_index), -1)
            length_tgt = (length_src * self.args.decode_upsample_scale).long().clip_(min=2)
        elif self.args.upsample_base == "predict":
            length_tgt = self.decoder.forward_length_prediction(
                self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
                encoder_out=encoder_out,
            )
            length_tgt = ((length_tgt-2) * self.args.decode_upsample_scale).long().clamp_(min=0) + 2
        elif self.args.upsample_base == "fixed":
            length_src = torch.sum(src_tokens.ne(self.src_dict.pad_index), -1)
            length_tgt = (length_src * 0 + self.args.decode_upsample_scale).long().clamp_(min=0) + 2
        else:
            raise NotImplementedError(f"Unknown upsample_base: {self.args.upsample_base}")

        if max_clip is not None and length_tgt.max() > max_clip:
            logging.warn("clip predicted length... Try a smaller validation batch size, or use bigger max_decoder_batch_tokens")
            length_tgt = length_tgt.clip(max=max_clip)

        initial_output_tokens = self.initialize_output_tokens_with_length(src_tokens, length_tgt)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0] if "encoder_out" in encoder_out else encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def max_positions(self):
        if vars(self.args).get("filter_max_length", None) is not None:
            if ":" not in self.args.filter_max_length:
                a = b = int(self.args.filter_max_length)
            else:
                a, b = self.args.filter_max_length.split(":")
                a, b = int(a), int(b)
            return (a, b)
        else:
            return (min(self.encoder.max_positions(), int(self.decoder.max_positions() / self.args.decode_upsample_scale)), self.decoder.max_positions())

    @torch.no_grad()
    def _analyze_graph(self, tgt_tokens, output_tokens, logits, links):
        tgt_tokens = tgt_tokens.long()
        target_length = (tgt_tokens != self.tgt_dict.pad_index).sum(dim=-1)
        output_length = (output_tokens != self.tgt_dict.pad_index).sum(dim=-1)
        batch_size, prelen, _ = links.shape

        # calculate node passing probability
        f_arr = []
        f_init = torch.zeros(batch_size, prelen, 1, dtype=links.dtype, device=links.device).fill_(float("-inf"))
        f_init[:, 0, 0].zero_()
        f_arr.append(f_init)
        for _ in range(1, prelen):
            f_now = torch.logsumexp(f_arr[-1] + links, 1, keepdim=True).transpose(1, 2) # batch * prelen * 1
            f_arr.append(f_now)
        f_arr = torch.cat(f_arr, -1).transpose(1, 2)
        node_pass_prob = f_arr.exp().sum(dim=1).tolist()

        # calculate max path
        from ..custom_ops import torch_dag_best_alignment, torch_dag_logsoftmax_gather_inplace
        word_ins_out, match = torch_dag_logsoftmax_gather_inplace(logits, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
        match = match.transpose(1, 2)
        paths = torch_dag_best_alignment(match, links, output_length, target_length).tolist()

        max_paths = []
        for i, raw_path in enumerate(paths):
            sample_max_paths = [-1 for _ in range(target_length[i])]
            for k, v in enumerate(raw_path):
                sample_max_paths[v] = k
            max_paths.append(sample_max_paths)

        # calculate top tokens
        top_k = 5
        val, idx = logits.softmax(dim=-1).topk(k=top_k, dim=-1)
        val = val.tolist()
        idx = idx.tolist()
        node_tokens = []
        node_probs = []
        for i in range(batch_size):
            sample_node_tokens = []
            sample_node_probs = []
            for j in range(output_length[i]):
                sample_node_tokens.append([self.tgt_dict[x] for x in idx[i][j]])
                sample_node_probs.append(val[i][j])
            node_tokens.append(sample_node_tokens)
            node_probs.append(sample_node_probs)

        links = torch.nan_to_num(links.softmax(dim=-1), nan=0).tolist()
        return {"node_pass_prob": node_pass_prob, "max_paths": max_paths, "node_tokens": node_tokens, "node_probs": node_probs, "links": links}


    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, decoding_graph=False, **kwargs):
        output_tokens = decoder_out.output_tokens
        rand_seed = random.randint(0, 19260817)

        bsz, seqlen = output_tokens.shape


        prev_output_tokens_position = (torch.arange(seqlen, dtype=torch.long, device=output_tokens.device).unsqueeze(0).expand(bsz, -1) + 1).\
                        masked_fill(output_tokens == self.tgt_dict.pad_index, 0)
        prev_output_tokens_segid = prev_output_tokens_position.masked_fill(output_tokens != self.tgt_dict.pad_index, 1).\
                        masked_fill_(output_tokens == self.tgt_dict.eos_index, 2)
        output_length = (output_tokens != self.tgt_dict.pad_index).sum(dim=-1) - 1
        bound_end = torch.zeros_like(prev_output_tokens_segid).masked_fill_(output_tokens != self.tgt_dict.pad_index, 1) * output_length.unsqueeze(-1)

        net_input = {
            "prev_output_tokens_segid": prev_output_tokens_segid,
            "bound_end": bound_end
        }

        output_logits, links = self.extract_features(output_tokens, encoder_out, net_input, rand_seed, require_links=True, training=False)

        if self.args.max_transition_length != -1:
            links = self.restore_valid_links(links)

        result = self.inference(decoder_out, output_logits, links)
        if not decoding_graph:
            return result

        if isinstance(result, DecodeResult):
            hypos_result = result.future.result()
            for fn, args in zip(result.fn, result.args):
                hypos_result = fn(hypos_result, *args)
            result = hypos_result
        return result, self._analyze_graph(result.output_tokens, output_tokens, output_logits, links)

    def inference_lookahead_repeatprevent(self, links, output_logits_normalized, output_length):

        batch_size, prelen, _ = links.shape

        top_logits, top_logits_idx = output_logits_normalized.topk(self.args.decode_top_cand_n, dim=-1)
        dagscores_arr = (links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
        dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n

        nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
        logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
        idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
        logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx] # batch * prelen * top_cand_n

        dagscores = dagscores.exp().cpu().numpy()
        nextstep_idx = nextstep_idx.int().cpu().numpy()
        logits_idx = logits_idx.int().cpu().numpy()
        output_length_cpu = output_length.int().cpu().numpy()

        output_tokens = []
        for i, length in enumerate(output_length_cpu):
            j = 0
            res = [top_logits_idx[i][0][0]]
            banned_ngram = set()
            while j != length - 1:
                temp_banned_token = set()
                if res:
                    temp_banned_token.add(res[-1])
                for k in range(2, min(self.args.decode_no_consecutive_repeated_ngram, (len(res) + 1) // 2)+1, 1):
                    if all([res[-l] == res[-k-l] for l in range(1, k, 1)]):
                        temp_banned_token.add(res[-k])
                prob = 0
                for k, cand in enumerate(logits_idx[i, j]):
                    if cand not in temp_banned_token and (tuple(res[-self.args.decode_no_repeated_ngram+1:]) + (cand, )) not in banned_ngram:
                        break
                    prob += dagscores[i, j, k]
                    if prob > self.args.decode_top_p:
                        k, cand = 0, logits_idx[i, j, 0]
                        break
                else:
                    k, cand = 0, logits_idx[i, j, 0]
                j = nextstep_idx[i, j, k]
                res.append(cand)
                if self.args.decode_no_repeated_ngram and len(res) >= self.args.decode_no_repeated_ngram:
                    banned_ngram.add(tuple(res[-self.args.decode_no_repeated_ngram:]))
            output_tokens.append(res)

        output_seqlen = max([len(res) for res in output_tokens])
        output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in output_tokens]
        output_tokens = torch.tensor(output_tokens)
        output_scores = torch.full(output_tokens.size(), 1.0)
        return output_tokens, output_scores

    def inference_lookahead_simple(self, links, output_logits_normalized, output_length):
        unreduced_logits, unreduced_tokens = output_logits_normalized.max(dim=-1)
        unreduced_tokens = unreduced_tokens.tolist()

        output_length = output_length.cpu().tolist()
        if self.args.decode_strategy == "lookahead":
            links_idx = (links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[1].cpu().tolist() # batch * prelen
        elif self.args.decode_strategy == "greedy":
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
        output_tokens = torch.tensor(output_tokens)
        output_scores = torch.full(output_tokens.size(), 1.0)
        return output_tokens, output_scores

    def inference_viterbi(self, links, output_logits_normalized, output_length):
        unreduced_logits, unreduced_tokens = output_logits_normalized.max(dim=-1)
        unreduced_tokens = unreduced_tokens.tolist()

        scores = []
        indexs = []
        # batch * graph_length
        alpha_t = links[:,0]
        if self.args.decode_strategy == "jointviterbi":
            alpha_t += unreduced_logits[:,0].unsqueeze(1)
        batch_size, graph_length, _ = links.size()
        alpha_t += unreduced_logits
        scores.append(alpha_t)

        # the exact max_length should be graph_length - 2, but we can reduce it to an appropriate extent to speedup decoding
        max_length = int(2 * graph_length / self.args.decode_upsample_scale)
        for i in range(max_length - 1):
            alpha_t, index = torch.max(alpha_t.unsqueeze(-1) + links, dim = 1)
            if self.args.decode_strategy == "jointviterbi":
                alpha_t += unreduced_logits
            scores.append(alpha_t)
            indexs.append(index)

        # max_length * batch * graph_length
        indexs = torch.stack(indexs, dim = 0)
        scores = torch.stack(scores, dim = 0)
        link_last = torch.gather(links, -1, (output_length - 1).view(batch_size, 1, 1).repeat(1, graph_length, 1)).view(1, batch_size, graph_length)
        scores += link_last

        # max_length * batch
        scores, max_idx = torch.max(scores, dim = -1)
        lengths = torch.arange(max_length).unsqueeze(-1).repeat(1, batch_size) + 1
        length_penalty = (lengths ** self.args.decode_viterbibeta).cuda(scores.get_device())
        scores = scores / length_penalty
        max_score, pred_length = torch.max(scores, dim = 0)
        pred_length = pred_length + 1

        initial_idx = torch.gather(max_idx, 0, (pred_length - 1).view(1, batch_size)).view(batch_size).tolist()
        unpad_output_tokens = []
        indexs = indexs.tolist()
        pred_length = pred_length.tolist()
        for i, length in enumerate(pred_length):
            j = initial_idx[i]
            last = unreduced_tokens[i][j]
            res = [last]
            for k in range(length - 1):
                j = indexs[length - k - 2][i][j]
                now_token = unreduced_tokens[i][j]
                if now_token != self.tgt_dict.pad_index and now_token != last:
                    res.insert(0, now_token)
                last = now_token
            unpad_output_tokens.append(res)

        output_seqlen = max([len(res) for res in unpad_output_tokens])
        output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
        output_tokens = torch.tensor(output_tokens)
        output_scores = torch.full(output_tokens.size(), 1.0)
        return output_tokens, output_scores

    def inference_sample(self, links, output_logits_normalized, output_length):
        batch_size, prelen, _ = links.shape

        top_logits, top_logits_idx = output_logits_normalized.topk(self.args.decode_top_cand_n, dim=-1)
        dagscores_arr = ((links / self.args.decode_temperature).log_softmax(dim=-1).unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
        dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n

        nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
        logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
        idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
        logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx] # batch * prelen * top_cand_n

        dagscores = dagscores.exp().cpu().numpy()
        nextstep_idx = nextstep_idx.int().cpu().numpy()
        logits_idx = logits_idx.int().cpu().numpy()
        output_length_cpu = output_length.int().cpu().numpy()

        output_tokens = []
        for i, length in enumerate(output_length_cpu):
            j = 0
            res = []
            banned_ngram = set()

            while j != length - 1:
                temp_banned_token = set()
                if res:
                    temp_banned_token.add(res[-1])
                for k in range(2, min(self.args.decode_no_consecutive_repeated_ngram, (len(res) + 1) // 2)+1, 1):
                    if all([res[-l] == res[-k-l] for l in range(1, k, 1)]):
                        temp_banned_token.add(res[-k])

                problist = []
                realprob = 0
                for k, cand in enumerate(logits_idx[i, j]):
                    realprob += dagscores[i, j, k]
                    if cand in temp_banned_token or (tuple(res[-self.args.decode_no_repeated_ngram+1:]) + (cand, )) in banned_ngram:
                        problist.append(1e-5)
                    else:
                        problist.append(dagscores[i, j, k])
                    if realprob > self.args.decode_top_p:
                        break
                problist = np.array(problist)
                problist /= problist.sum()
                k = np.random.choice(len(problist), 1, p=problist).item()
                cand = logits_idx[i, j, k]

                j = nextstep_idx[i, j, k]
                res.append(cand)

                if self.args.decode_no_repeated_ngram and len(res) >= self.args.decode_no_repeated_ngram:
                    banned_ngram.add(tuple(res[-self.args.decode_no_repeated_ngram:]))

            output_tokens.append(res)

        output_seqlen = max([len(res) for res in output_tokens])
        output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in output_tokens]
        output_tokens = torch.tensor(output_tokens)
        output_scores = torch.full(output_tokens.size(), 1.0)
        return output_tokens, output_scores

    def inference_beamsearch(self, links, output_logits_normalized, output_length):
        batch_size, prelen, _ = links.shape

        assert batch_size <= self.args.decode_max_batchsize, "Please set --decode-max-batchsize for beamsearch with a larger batch size"

        top_logits, top_logits_idx = output_logits_normalized.topk(self.args.decode_top_cand_n, dim=-1)
        dagscores_arr = (links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
        dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n

        nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
        logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
        idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
        logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx] # batch * prelen * top_cand_n

        # rearange_idx = logits_idx.sort(dim=-1)[1]
        # dagscores = dagscores.gather(-1, rearange_idx) # batch * prelen * top_cand_n
        # nextstep_idx = nextstep_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
        # logits_idx = logits_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n

        if dagscores.get_device() == -1 and self.args.decode_strategy == "beamsearch" and self.args.decode_max_workers < 1:
            raise RuntimeError("Please specify decode_max_workers at least 1 if you want to run DA-Transformer on cpu while using beamsearch decoding. "
                               "It will use a seperate process for beamsearch because the multi-thread library used in PyTorch and DAG-Search is conflict.")

        dagscores = np.ascontiguousarray(dagscores.float().cpu().numpy())
        nextstep_idx = np.ascontiguousarray(nextstep_idx.int().cpu().numpy())
        logits_idx = np.ascontiguousarray(logits_idx.int().cpu().numpy())
        output_length_cpu = np.ascontiguousarray(output_length.int().cpu().numpy())

        if self.args.decode_max_workers >= 1:
            future = self.executor.submit(call_dag_search, dagscores, nextstep_idx, logits_idx,
                output_length_cpu,
                self.args.decode_alpha,
                self.args.decode_gamma,
                self.args.decode_beamsize,
                self.args.decode_max_beam_per_length,
                self.args.decode_top_p,
                self.tgt_dict.pad_index,
                self.tgt_dict.bos_index,
                1 if self.args.decode_dedup else 0,
                self.args.decode_no_consecutive_repeated_ngram,
                self.args.decode_no_repeated_ngram)
            return future
        else:
            res = call_dag_search(dagscores, nextstep_idx, logits_idx,
                output_length_cpu,
                self.args.decode_alpha,
                self.args.decode_gamma,
                self.args.decode_beamsize,
                self.args.decode_max_beam_per_length,
                self.args.decode_top_p,
                self.tgt_dict.pad_index,
                self.tgt_dict.bos_index,
                1 if self.args.decode_dedup else 0,
                self.args.decode_no_consecutive_repeated_ngram,
                self.args.decode_no_repeated_ngram)
            return res

    def inference(self, decoder_out, output_logits, links):
        output_tokens = decoder_out.output_tokens
        output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)

        output_logits_normalized = output_logits.log_softmax(dim=-1)

        if self.args.decode_strategy in ["lookahead", "greedy"]:
            if self.args.decode_no_consecutive_repeated_ngram > 0 or self.args.decode_no_repeated_ngram > 0 and self.args.decode_strategy == 'lookahead':
                inference_result = self.inference_lookahead_repeatprevent(links, output_logits_normalized, output_length)
            else:
                inference_result = self.inference_lookahead_simple(links, output_logits_normalized, output_length)
        elif self.args.decode_strategy in ["viterbi", "jointviterbi"]:
            assert self.args.decode_no_consecutive_repeated_ngram == 0 and self.args.decode_no_repeated_ngram == 0, "viterbi decoding does not support repeated ngram prevention"
            inference_result = self.inference_viterbi(links, output_logits_normalized, output_length)
        elif self.args.decode_strategy == "sample":
            inference_result = self.inference_sample(links, output_logits_normalized, output_length)
        elif self.args.decode_strategy == "beamsearch":
            inference_result = self.inference_beamsearch(links, output_logits_normalized, output_length)

        if isinstance(inference_result, concurrent.futures.Future):
            return DecodeResult(future=inference_result, fn=[inference_post_process], args=[(decoder_out, )])
        else:
            return inference_post_process(inference_result, decoder_out)

def inference_post_process(inference_result, decoder_out): # post process after inference for overlapped decoding
    output_tokens, output_scores = inference_result
    return decoder_out._replace(
        output_tokens=output_tokens,
        output_scores=output_scores,
        attn=None,
        history=None,
    )

class GlatLinkDecoder(NATransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.init_link_feature(args)
        if self.args.segment_embedding:
            self.init_seg_feature(args, dictionary)

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

    def init_seg_feature(self, args, dictionary):
        self.embed_seg = PositionalEmbedding(dictionary.last_seg_token - dictionary.first_seg_token, args.decoder_embed_dim, 0, True)

    def forward_embedding(self, prev_output_tokens, prev_output_tokens_segid):
        assert prev_output_tokens.shape[1] == prev_output_tokens_segid.shape[1]

        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.args.segment_embedding:
            x += F.embedding(prev_output_tokens_segid, self.embed_seg.weight, self.embed_seg.padding_idx,
                    self.embed_seg.max_norm, self.embed_seg.norm_type, self.embed_seg.scale_grad_by_freq, self.embed_seg.sparse)

        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def extract_features(
        self,
        prev_output_tokens,
        net_input,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        assert embedding_copy is False
        x, decoder_padding_mask = self.forward_embedding(prev_output_tokens, \
                net_input['prev_output_tokens_segid'])

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

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

@register_model_architecture(
    "glat_decomposed_link", "glat_decomposed_link_pretrain"
)
def base_architecture_pretrain(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    base_architecture(args)

@register_model_architecture(
    "glat_decomposed_link", "glat_decomposed_link_big"
)
def big_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)
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
    MultiheadAttention
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.nat.nonautoregressive_transformer import NATransformerDecoder
from .ls_transformer import LSTransformerModel, LSFSTransformerDecoderLayer, LSTransformerEncoder
from .ls_nat_decoder import LSNATransformerDecoder

try:
    from lightseq.training.ops.pytorch.transformer_embedding_layer import (
    LSTransformerEmbeddingLayer,
    )
    from lightseq.training.ops.pytorch.transformer_encoder_layer import (
        LSTransformerEncoderLayer,
    )
except ModuleNotFoundError as err:
    pass

from .glat_decomposed_with_link import GlatDecomposedLink, GlatLinkDecoder, torch_seed


logger = logging.getLogger(__name__)

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)
    if isinstance(module, LSTransformerEmbeddingLayer):
        normal_(module.embeddings.data)
    if isinstance(module, LSTransformerEncoderLayer):
        normal_(module._get_weights(0).data)
        module._get_weights(1).data.zero_()
        normal_(module._get_weights(2).data)
        module._get_weights(3).data.zero_()
        normal_(module._get_weights(6).data)
        module._get_weights(7).data.zero_()
        normal_(module._get_weights(8).data)
        module._get_weights(9).data.zero_()
    if isinstance(module, LSFSTransformerDecoderLayer):
        normal_(module._get_weights(0).data)
        module._get_weights(1).data.zero_()
        normal_(module._get_weights(2).data)
        module._get_weights(3).data.zero_()
        normal_(module._get_weights(6).data)
        module._get_weights(7).data.zero_()
        normal_(module._get_weights(8).data)
        module._get_weights(9).data.zero_()
        normal_(module._get_weights(12).data)
        module._get_weights(13).data.zero_()
        normal_(module._get_weights(14).data)
        module._get_weights(15).data.zero_()
        if module.config.layer_id == 0:
            normal_(module._get_weights(18).data)

# @jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))

@register_model("ls_glat_decomposed_link")
class LSGlatDecomposedLink(LSTransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.tgt_dict = decoder.dictionary
        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()

        self.ensemble_models = None
        self.init_beam_search()

    init_beam_search = GlatDecomposedLink.init_beam_search

    @property
    def allow_length_beam(self):
        return False

    @property
    def allow_ensemble(self):
        return False

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LSGlatLinkDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = LSTransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    add_args = GlatDecomposedLink.add_args

    extract_valid_links = GlatDecomposedLink.extract_valid_links
    restore_valid_links = GlatDecomposedLink.restore_valid_links
    extract_links = GlatDecomposedLink.extract_links
    extract_features = GlatDecomposedLink.extract_features
    forward = GlatDecomposedLink.forward
    initialize_output_tokens_with_length = GlatDecomposedLink.initialize_output_tokens_with_length
    initialize_output_tokens_upsample_by_tokens = GlatDecomposedLink.initialize_output_tokens_upsample_by_tokens
    initialize_output_tokens_multiplier_by_tokens = GlatDecomposedLink.initialize_output_tokens_multiplier_by_tokens
    initialize_output_tokens_by_tokens = GlatDecomposedLink.initialize_output_tokens_by_tokens
    initialize_output_tokens = GlatDecomposedLink.initialize_output_tokens
    max_positions = GlatDecomposedLink.max_positions
    forward_decoder = GlatDecomposedLink.forward_decoder
    initialize_output_tokens = GlatDecomposedLink.initialize_output_tokens

    def forward_encoder(self, encoder_inputs):
        return self.encoder(encoder_inputs[0])
   
    def initialize_output_tokens_upsample(self, encoder_out, src_tokens):
        # length prediction
        if vars(self.args).get("src_upsample_scale", None) is not None:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            length_tgt = (length_tgt * self.args.src_upsample_scale).long().clamp_(min=2)
        else:
            assert self.args.src_upsample_fixed is not None, "Must specify --src-upsample-scale or --src-upsample-fixed"
            length_tgt = torch.zeros(src_tokens.shape[0], device=src_tokens.device, dtype=src_tokens.dtype).fill_(self.args.src_upsample_fixed)
        initial_output_tokens = self.initialize_output_tokens_with_length(src_tokens, length_tgt)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

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
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

class LSGlatLinkDecoder(LSNATransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.init_link_feature(args)

    init_link_feature = GlatLinkDecoder.init_link_feature

    def build_decoder_layer(self, args):
        if args.max_decoder_batch_tokens is not None:
            max_batch_tokens = args.max_decoder_batch_tokens
        elif args.src_upsample_scale is not None:
            max_batch_tokens = int(args.max_tokens * args.src_upsample_scale)
        else:
            raise RuntimeError("Must specify --max-decoder-batch-tokens if no --src_upsample_scale is not specified when using Lightseq Decoder")

        config = LSFSTransformerDecoderLayer.get_config(
            max_batch_tokens=max_batch_tokens,
            max_seq_len=args.max_target_positions,
            hidden_size=args.decoder_embed_dim,
            intermediate_size=args.decoder_ffn_embed_dim,
            nhead=args.decoder_attention_heads,
            attn_prob_dropout_ratio=args.attention_dropout,
            activation_dropout_ratio=args.activation_dropout,
            hidden_dropout_ratio=args.dropout,
            pre_layer_norm=args.decoder_normalize_before,
            fp16=args.fp16,
            local_rank=args.device_id,
            nlayer=args.decoder_layers,
            activation_fn=args.activation_fn,
        )
        return LSFSTransformerDecoderLayer(config)

    @staticmethod
    def add_args(parser):
        pass

@register_model_architecture(
    "ls_glat_decomposed_link", "ls_glat_decomposed_link_6e6d512"
)
def base_architecture(args):
    args.max_source_positions = getattr(args, "max_source_positions", 300)
    args.max_target_positions = getattr(args, "max_target_positions", 300)


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
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)

@register_model_architecture(
    "ls_glat_decomposed_link", "ls_glat_decomposed_link_base"
)
def base_architecture2(args):
    base_architecture(args)

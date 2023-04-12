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

from .glat_decomposed_with_link import GlatDecomposedLink, GlatLinkDecoder


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

@register_model("ls_glat_decomposed_link")
class LSGlatDecomposedLink(LSTransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.src_dict = encoder.dictionary
        self.tgt_dict = decoder.dictionary
        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()
        self.ensemble_models = None
        self.fast_generate = False
        self.init_beam_search()

    init_beam_search = GlatDecomposedLink.init_beam_search

    @property
    def allow_length_beam(self):
        return False

    @property
    def allow_ensemble(self):
        return False

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
                    logging.warn("--segment_embedding disabled. dropping decoder.embed_seg.weight ...")
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

            if "encoder.layers.0.fc1.bias" in states.keys():
                logging.warn("Automatically converting fairseq transformer to lightseq transformer. "
                    "Please make sure that the architecture is matched; otherwise it may cause unexpected behaviours.")
                from ..scripts._convert_utils import convert_state_from_fs_to_ls
                states = convert_state_from_fs_to_ls(states, args.decoder_embed_dim, args.decoder_ffn_embed_dim, args.decoder_layers)

            model.load_state_dict(states)
            args.load_pretrained_model = None  # Clear this param

        return model

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
    max_positions = GlatDecomposedLink.max_positions
    forward_decoder = GlatDecomposedLink.forward_decoder
    initialize_output_tokens_with_length = GlatDecomposedLink.initialize_output_tokens_with_length
    initialize_output_tokens = GlatDecomposedLink.initialize_output_tokens
    inference_lookahead_repeatprevent = GlatDecomposedLink.inference_lookahead_repeatprevent
    inference_lookahead_simple = GlatDecomposedLink.inference_lookahead_simple
    inference_viterbi = GlatDecomposedLink.inference_viterbi
    inference_sample = GlatDecomposedLink.inference_sample
    inference_beamsearch = GlatDecomposedLink.inference_beamsearch
    inference = GlatDecomposedLink.inference

    def forward_encoder(self, encoder_inputs):
        return self.encoder(encoder_inputs[0])

class LSGlatLinkDecoder(LSNATransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.init_link_feature(args)
        if self.args.segment_embedding:
            self.init_seg_feature(args, dictionary)
        self.project_in_dim = None

    init_link_feature = GlatLinkDecoder.init_link_feature
    init_seg_feature = GlatLinkDecoder.init_seg_feature

    def build_decoder_layer(self, args):
        if args.max_decoder_batch_tokens is not None:
            max_batch_tokens = args.max_decoder_batch_tokens
        else:
            raise RuntimeError("Must specify --max-decoder-batch-tokens when using Lightseq Decoder")

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

    forward_embedding = GlatLinkDecoder.forward_embedding

    def extract_features(
        self,
        prev_output_tokens,
        net_input,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        incremental_state=None,
        **unused
    ):
        # x, prev_output_tokens = self.forward_embedding(
        #     prev_output_tokens, incremental_state
        # )
        x, decoder_padding_mask = self.forward_embedding(prev_output_tokens, \
                net_input['prev_output_tokens_segid'])

        # x: [batch_size, seq_len, hidden_size]
        for _, layer in enumerate(self.layers):
            x, _, _ = layer(
                x,
                decoder_padding_mask,
                encoder_out.encoder_out,
                encoder_out.encoder_padding_mask,
                incremental_state,
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x, {}

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

@register_model_architecture(
    "ls_glat_decomposed_link", "ls_glat_decomposed_link_pretrain"
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
    "ls_glat_decomposed_link", "ls_glat_decomposed_link_big"
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
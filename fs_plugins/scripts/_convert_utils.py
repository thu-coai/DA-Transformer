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

import re
import torch
from torch import nn

def calc_offset(sizes):
    offsets = [0]
    tmp = 0
    for x in sizes:
        tmp += x
        offsets.append(tmp)
    return offsets

_shared_encdec_attn_kv_params = dict()

class DummyLSDecoder(nn.Module):
    def __init__(self, para, hidden_size, intermediate_size, nlayer, layer_id):
        super().__init__()

        self.para_offset = DummyLSDecoder.gen_offset(
            hidden_size, intermediate_size, nlayer
        )
        self.layer_id = layer_id
        self.hidden_size = hidden_size

        if layer_id != 0:
            self.para_offset = self.para_offset[:-2]
        if para is None:
            self.para = torch.Tensor(self.para_offset[-1])
        else:
            self.para = para

    @staticmethod
    def gen_offset(hidden_size, intermediate_size, nlayer):
        hs, ims = hidden_size, intermediate_size
        sizes = [
            hs * hs * 3,  # attn_qkvw
            hs * 3,  # attn_qkvb
            hs * hs,  # attn_ow
            hs,  # attn_ob
            hs,  # attn_nw
            hs,  # attn_nb
            hs * hs,  # encdec_attn_qw
            hs,  # encdec_attn_qb
            hs * hs,  # encdec_attn_ow
            hs,  # encdec_attn_ob
            hs,  # encdec_attn_nw
            hs,  # encdec_attn_nb
            hs * ims,  # inter_w
            ims,  # inter_b
            hs * ims,  # output_w
            hs,  # output_b
            hs,  # ffn_nw
            hs,  # ffn_nb
            hs * hs * 2 * nlayer,  # encdec_attn_kvw
            hs * 2 * nlayer,  # encdec_attn_kvb
        ]
        offsets = calc_offset(sizes)
        return offsets

    def _get_weights(self, i):
        return self.para.data.narrow(
            0, self.para_offset[i], self.para_offset[i + 1] - self.para_offset[i]
        )

    def split_weights(self):
        weights = [self._get_weights(i) for i in range(18)]

        hs = self.hidden_size
        idx = self.layer_id

        if idx == 0:
            _shared_encdec_attn_kv_params["w"] = self._get_weights(18)
            _shared_encdec_attn_kv_params["b"] = self._get_weights(19)
        encdec_kvw = _shared_encdec_attn_kv_params["w"]
        encdec_kvb = _shared_encdec_attn_kv_params["b"]

        offset = hs * hs * 2 * idx
        encdec_kvw = encdec_kvw.data.narrow(0, offset, hs * hs * 2)
        offset = hs * 2 * idx
        encdec_kvb = encdec_kvb.data.narrow(0, offset, hs * 2)
        weights += [encdec_kvw, encdec_kvb]
        weights[0] = weights[0].view(-1, hs)
        weights[2] = weights[2].view(-1, hs)
        weights[6] = weights[6].view(-1, hs)
        weights[8] = weights[8].view(-1, hs)
        weights[12] = weights[12].view(-1, hs)
        weights[14] = weights[14].view(hs, -1)
        weights[18] = weights[18].view(-1, hs)
        return weights

class DummyLSEncoder(nn.Module):
    def __init__(self, para, hidden_size, intermediate_size):
        super().__init__()

        self.para_offset = DummyLSEncoder.gen_offset(
            hidden_size, intermediate_size
        )
        if para is None:
            self.para = torch.Tensor(self.para_offset[-1])
        else:
            self.para = para

    @staticmethod
    def gen_offset(hidden_size, intermediate_size):
        hs, ims = hidden_size, intermediate_size
        sizes = [
            hs * hs * 3,  # attn_qkvw
            hs * 3,  # attn_qkvb
            hs * hs,  # attn_ow
            hs,  # attn_ob
            hs,  # attn_nw
            hs,  # attn_nb
            hs * ims,  # inter_w
            ims,  # inter_b
            hs * ims,  # output_w
            hs,  # output_b
            hs,  # ffn_nw
            hs,  # ffn_nb
        ]
        offsets = calc_offset(sizes)
        return offsets

    def _get_weights(self, i):
        return self.para.data.narrow(
            0, self.para_offset[i], self.para_offset[i + 1] - self.para_offset[i]
        )

def convert_state_from_ls_to_fs(state_dict, hidden_size, intermediate_size, nlayer):
    new_params = {}

    for key, param in state_dict.items():
        m = re.match(r"decoder.layers.(\d+).para", key)
        if m is not None:
            idx = int(m.group(1))
            layer = DummyLSDecoder(param, hidden_size, intermediate_size, nlayer, idx)
            attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb, \
                encdec_attn_qw, encdec_attn_qb, encdec_attn_ow, encdec_attn_ob, encdec_attn_nw, encdec_attn_nb, \
                inter_w, inter_b, output_w, output_b, ffn_nw, ffn_nb, \
                encdec_attn_kvw, encdec_attn_kvb = layer.split_weights()
            attn_qb, attn_kb, attn_vb = attn_qkvb.chunk(3, dim=0)
            attn_qw, attn_kw, attn_vw = attn_qkvw.chunk(3, dim=0)
            attn_qb, attn_kb, attn_vb = attn_qkvb.chunk(3, dim=0)
            encdec_attn_kw, encdec_attn_vw = encdec_attn_kvw.chunk(2, dim=0)
            encdec_attn_kb, encdec_attn_vb = encdec_attn_kvb.chunk(2, dim=0)
            root = f"decoder.layers.{idx}"
            new_params[f'{root}.encoder_attn.k_proj.bias'] = encdec_attn_kb
            new_params[f'{root}.encoder_attn.k_proj.weight'] = encdec_attn_kw
            new_params[f'{root}.encoder_attn.out_proj.bias'] = encdec_attn_ob
            new_params[f'{root}.encoder_attn.out_proj.weight'] = encdec_attn_ow
            new_params[f'{root}.encoder_attn.q_proj.bias'] = encdec_attn_qb
            new_params[f'{root}.encoder_attn.q_proj.weight'] = encdec_attn_qw
            new_params[f'{root}.encoder_attn.v_proj.bias'] = encdec_attn_vb
            new_params[f'{root}.encoder_attn.v_proj.weight'] = encdec_attn_vw
            new_params[f'{root}.encoder_attn_layer_norm.bias'] = encdec_attn_nb
            new_params[f'{root}.encoder_attn_layer_norm.weight'] = encdec_attn_nw
            new_params[f'{root}.fc1.bias'] = inter_b
            new_params[f'{root}.fc1.weight'] = inter_w
            new_params[f'{root}.fc2.bias'] = output_b
            new_params[f'{root}.fc2.weight'] = output_w
            new_params[f'{root}.final_layer_norm.bias'] = ffn_nb
            new_params[f'{root}.final_layer_norm.weight'] = ffn_nw
            new_params[f'{root}.self_attn.k_proj.bias'] = attn_kb
            new_params[f'{root}.self_attn.k_proj.weight'] = attn_kw
            new_params[f'{root}.self_attn.out_proj.bias'] = attn_ob
            new_params[f'{root}.self_attn.out_proj.weight'] = attn_ow
            new_params[f'{root}.self_attn.q_proj.bias'] = attn_qb
            new_params[f'{root}.self_attn.q_proj.weight'] = attn_qw
            new_params[f'{root}.self_attn.v_proj.bias'] = attn_vb
            new_params[f'{root}.self_attn.v_proj.weight'] = attn_vw
            new_params[f'{root}.self_attn_layer_norm.bias'] = attn_nb
            new_params[f'{root}.self_attn_layer_norm.weight'] = attn_nw
            continue

        m = re.match(r"encoder.layers.(\d+).para", key)
        if m is not None:
            idx = int(m.group(1))
            layer = DummyLSEncoder(param, hidden_size, intermediate_size)
            hs = hidden_size
            attn_qkvw = layer._get_weights(0).view(-1, hs)
            attn_qw, attn_kw, attn_vw = attn_qkvw.chunk(3, dim=0)
            attn_qkvb = layer._get_weights(1)
            attn_qb, attn_kb, attn_vb = attn_qkvb.chunk(3, dim=0)
            attn_ow = layer._get_weights(2).view(-1, hs)
            attn_ob = layer._get_weights(3)
            attn_nw = layer._get_weights(4)
            attn_nb = layer._get_weights(5)
            inter_w = layer._get_weights(6).view(-1, hs)
            inter_b = layer._get_weights(7)
            output_w = layer._get_weights(8).view(hs, -1)
            output_b = layer._get_weights(9)
            ffn_nw = layer._get_weights(10)
            ffn_nb = layer._get_weights(11)
            root = f"encoder.layers.{idx}"
            new_params[f'{root}.fc1.bias'] = inter_b
            new_params[f'{root}.fc1.weight'] = inter_w
            new_params[f'{root}.fc2.bias'] = output_b
            new_params[f'{root}.fc2.weight'] = output_w
            new_params[f'{root}.final_layer_norm.bias'] = ffn_nb
            new_params[f'{root}.final_layer_norm.weight'] = ffn_nw
            new_params[f'{root}.self_attn.k_proj.bias'] = attn_kb
            new_params[f'{root}.self_attn.k_proj.weight'] = attn_kw
            new_params[f'{root}.self_attn.out_proj.bias'] = attn_ob
            new_params[f'{root}.self_attn.out_proj.weight'] = attn_ow
            new_params[f'{root}.self_attn.q_proj.bias'] = attn_qb
            new_params[f'{root}.self_attn.q_proj.weight'] = attn_qw
            new_params[f'{root}.self_attn.v_proj.bias'] = attn_vb
            new_params[f'{root}.self_attn.v_proj.weight'] = attn_vw
            new_params[f'{root}.self_attn_layer_norm.bias'] = attn_nb
            new_params[f'{root}.self_attn_layer_norm.weight'] = attn_nw
            continue

        new_params[key] = param
    return new_params

def convert_state_from_fs_to_ls(state_dict, hidden_size, intermediate_size, nlayer):
    new_params = {}
    encoder_layer = []
    decoder_layer = []

    for idx in range(nlayer):
        layer = DummyLSEncoder(None, hidden_size, intermediate_size)
        encoder_layer.append(layer)
        hs = hidden_size
        attn_qkvw = layer._get_weights(0).view(-1, hs)
        attn_qw, attn_kw, attn_vw = attn_qkvw.chunk(3, dim=0)
        attn_qkvb = layer._get_weights(1)
        attn_qb, attn_kb, attn_vb = attn_qkvb.chunk(3, dim=0)
        attn_ow = layer._get_weights(2).view(-1, hs)
        attn_ob = layer._get_weights(3)
        attn_nw = layer._get_weights(4)
        attn_nb = layer._get_weights(5)
        inter_w = layer._get_weights(6).view(-1, hs)
        inter_b = layer._get_weights(7)
        output_w = layer._get_weights(8).view(hs, -1)
        output_b = layer._get_weights(9)
        ffn_nw = layer._get_weights(10)
        ffn_nb = layer._get_weights(11)
        root = f"encoder.layers.{idx}"
        inter_b.copy_(state_dict[f'{root}.fc1.bias'])
        inter_w.copy_(state_dict[f'{root}.fc1.weight'])
        output_b.copy_(state_dict[f'{root}.fc2.bias'])
        output_w.copy_(state_dict[f'{root}.fc2.weight'])
        ffn_nb.copy_(state_dict[f'{root}.final_layer_norm.bias'])
        ffn_nw.copy_(state_dict[f'{root}.final_layer_norm.weight'])
        attn_kb.copy_(state_dict[f'{root}.self_attn.k_proj.bias'])
        attn_kw.copy_(state_dict[f'{root}.self_attn.k_proj.weight'])
        attn_ob.copy_(state_dict[f'{root}.self_attn.out_proj.bias'])
        attn_ow.copy_(state_dict[f'{root}.self_attn.out_proj.weight'])
        attn_qb.copy_(state_dict[f'{root}.self_attn.q_proj.bias'])
        attn_qw.copy_(state_dict[f'{root}.self_attn.q_proj.weight'])
        attn_vb.copy_(state_dict[f'{root}.self_attn.v_proj.bias'])
        attn_vw.copy_(state_dict[f'{root}.self_attn.v_proj.weight'])
        attn_nb.copy_(state_dict[f'{root}.self_attn_layer_norm.bias'])
        attn_nw.copy_(state_dict[f'{root}.self_attn_layer_norm.weight'])


    for idx in range(nlayer):
        layer = DummyLSDecoder(None, hidden_size, intermediate_size, nlayer, idx)
        decoder_layer.append(layer)
        attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb, \
            encdec_attn_qw, encdec_attn_qb, encdec_attn_ow, encdec_attn_ob, encdec_attn_nw, encdec_attn_nb, \
            inter_w, inter_b, output_w, output_b, ffn_nw, ffn_nb, \
            encdec_attn_kvw, encdec_attn_kvb = layer.split_weights()
        attn_qb, attn_kb, attn_vb = attn_qkvb.chunk(3, dim=0)
        attn_qw, attn_kw, attn_vw = attn_qkvw.chunk(3, dim=0)
        attn_qb, attn_kb, attn_vb = attn_qkvb.chunk(3, dim=0)
        encdec_attn_kw, encdec_attn_vw = encdec_attn_kvw.chunk(2, dim=0)
        encdec_attn_kb, encdec_attn_vb = encdec_attn_kvb.chunk(2, dim=0)
        root = f"decoder.layers.{idx}"
        encdec_attn_kb.copy_(state_dict[f'{root}.encoder_attn.k_proj.bias'])
        encdec_attn_kw.copy_(state_dict[f'{root}.encoder_attn.k_proj.weight'])
        encdec_attn_ob.copy_(state_dict[f'{root}.encoder_attn.out_proj.bias'])
        encdec_attn_ow.copy_(state_dict[f'{root}.encoder_attn.out_proj.weight'])
        encdec_attn_qb.copy_(state_dict[f'{root}.encoder_attn.q_proj.bias'])
        encdec_attn_qw.copy_(state_dict[f'{root}.encoder_attn.q_proj.weight'])
        encdec_attn_vb.copy_(state_dict[f'{root}.encoder_attn.v_proj.bias'])
        encdec_attn_vw.copy_(state_dict[f'{root}.encoder_attn.v_proj.weight'])
        encdec_attn_nb.copy_(state_dict[f'{root}.encoder_attn_layer_norm.bias'])
        encdec_attn_nw.copy_(state_dict[f'{root}.encoder_attn_layer_norm.weight'])
        inter_b.copy_(state_dict[f'{root}.fc1.bias'])
        inter_w.copy_(state_dict[f'{root}.fc1.weight'])
        output_b.copy_(state_dict[f'{root}.fc2.bias'])
        output_w.copy_(state_dict[f'{root}.fc2.weight'])
        ffn_nb.copy_(state_dict[f'{root}.final_layer_norm.bias'])
        ffn_nw.copy_(state_dict[f'{root}.final_layer_norm.weight'])
        attn_kb.copy_(state_dict[f'{root}.self_attn.k_proj.bias'])
        attn_kw.copy_(state_dict[f'{root}.self_attn.k_proj.weight'])
        attn_ob.copy_(state_dict[f'{root}.self_attn.out_proj.bias'])
        attn_ow.copy_(state_dict[f'{root}.self_attn.out_proj.weight'])
        attn_qb.copy_(state_dict[f'{root}.self_attn.q_proj.bias'])
        attn_qw.copy_(state_dict[f'{root}.self_attn.q_proj.weight'])
        attn_vb.copy_(state_dict[f'{root}.self_attn.v_proj.bias'])
        attn_vw.copy_(state_dict[f'{root}.self_attn.v_proj.weight'])
        attn_nb.copy_(state_dict[f'{root}.self_attn_layer_norm.bias'])
        attn_nw.copy_(state_dict[f'{root}.self_attn_layer_norm.weight'])

    for idx in range(nlayer):
        new_params[f'decoder.layers.{idx}.para'] = decoder_layer[idx].para
        new_params[f'encoder.layers.{idx}.para'] = encoder_layer[idx].para

    for key, param in state_dict.items():
        if not key.startswith('encoder.layers.') and not key.startswith('decoder.layers.'):
            new_params[key] = param

    return new_params

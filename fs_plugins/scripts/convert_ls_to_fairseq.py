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

import argparse
import collections
import os
import re
import logging

import torch
from fairseq.file_io import PathManager
from fairseq import checkpoint_utils, utils


logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
logger = logging.getLogger("convert_ls_to_fairseq")


def convert_checkpoint(input):
    params_dict = collections.OrderedDict()

    logger.info("loading model(s) from {}".format(input))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        [input],
        arg_overrides={"max_tokens":128}
    )
    model = models[0]

    with PathManager.open(input, "rb") as f:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )

    assert state['cfg']['model']._name.startswith("ls_glat_decomposed_link"), "The script only converts ls_glat_decomposed_link to glat_decomposed_link"
    state['cfg']['model']._name = state['cfg']['model']._name[3:]
    state['last_optimizer_state'] = {}  # drop optimizer state

    model_params = state["model"]
    state["model"] = new_params = {}
    for key, param in model_params.items():
        m = re.match(r"decoder.layers.(\d+).para", key)
        if m is not None:
            idx = int(m.group(1))
            attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb, \
                encdec_attn_qw, encdec_attn_qb, encdec_attn_ow, encdec_attn_ob, encdec_attn_nw, encdec_attn_nb, \
                inter_w, inter_b, output_w, output_b, ffn_nw, ffn_nb, \
                encdec_attn_kvw, encdec_attn_kvb = model.decoder.layers[idx].split_weights()
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
            hs = model.encoder.layers[idx].config.hidden_size
            attn_qkvw = model.encoder.layers[idx]._get_weights(0).view(-1, hs)
            attn_qw, attn_kw, attn_vw = attn_qkvw.chunk(3, dim=0)
            attn_qkvb = model.encoder.layers[idx]._get_weights(1)
            attn_qb, attn_kb, attn_vb = attn_qkvb.chunk(3, dim=0)
            attn_ow = model.encoder.layers[idx]._get_weights(2).view(-1, hs)
            attn_ob = model.encoder.layers[idx]._get_weights(3)
            attn_nw = model.encoder.layers[idx]._get_weights(4)
            attn_nb = model.encoder.layers[idx]._get_weights(5)
            inter_w = model.encoder.layers[idx]._get_weights(6).view(-1, hs)
            inter_b = model.encoder.layers[idx]._get_weights(7)
            output_w = model.encoder.layers[idx]._get_weights(8).view(hs, -1)
            output_b = model.encoder.layers[idx]._get_weights(9)
            ffn_nw = model.encoder.layers[idx]._get_weights(10)
            ffn_nb = model.encoder.layers[idx]._get_weights(11)
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

    return state

def main():
    parser = argparse.ArgumentParser(
        description="Convert a DA-Transformer trained with ls_glat_decomposed_link to glat_decomposed_link",
    )
    # fmt: off
    parser.add_argument('--input', required=True, help='Input checkpoint file path.')
    parser.add_argument('--output', required=True, help='Output checkpoint file path.')
    parser.add_argument('--user-dir', default="./fs_plugins")
    parser.add_argument('--debug', action="store_true")
    # fmt: on
    args = parser.parse_args()
    print(args)

    if args.debug:
        import ptvsd
        ptvsd.enable_attach()
        import logging
        logging.warning("wait debug")
        ptvsd.wait_for_attach()

    utils.import_user_module(args)
    new_state = convert_checkpoint(args.input)
    with PathManager.open(args.output, "wb") as f:
        torch.save(new_state, f)
    print("Finished writing converted checkpoint to {}".format(args.output))

if __name__ == "__main__":
    main()

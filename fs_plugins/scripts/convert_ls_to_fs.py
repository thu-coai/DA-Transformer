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
import os
import logging

import torch
from fairseq.file_io import PathManager

from _convert_utils import convert_state_from_ls_to_fs

logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
logger = logging.getLogger("convert_ls_to_fs")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a DA-Transformer trained with ls_glat_decomposed_link to glat_decomposed_link",
    )
    # fmt: off
    parser.add_argument('--input', required=True, help='Input checkpoint file path.')
    parser.add_argument('--output', required=True, help='Output checkpoint file path.')
    parser.add_argument('--debug', action="store_true")
    # fmt: on
    args = parser.parse_args()
    print(args)

    if args.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", 5679))
        logging.info("wait debug")
        debugpy.wait_for_client()

    state = torch.load(args.input, map_location='cpu')
    assert state['cfg']['model']._name.startswith("ls_glat_decomposed_link"), "The script only converts ls_glat_decomposed_link to glat_decomposed_link"
    state['cfg']['model']._name = state['cfg']['model']._name[3:]
    state['last_optimizer_state'] = {}  # drop optimizer state

    model_params = state["model"]
    hidden_size = state["cfg"]['model'].decoder_embed_dim
    intermediate_size = state["cfg"]['model'].decoder_ffn_embed_dim
    nlayer = state["cfg"]['model'].decoder_layers
    state["model"] = convert_state_from_ls_to_fs(model_params, hidden_size, intermediate_size, nlayer)

    with PathManager.open(args.output, "wb") as f:
        torch.save(state, f)
    print("Finished writing converted checkpoint to {}".format(args.output))

if __name__ == "__main__":
    main()

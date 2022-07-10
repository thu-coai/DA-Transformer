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

# %%
import argparse
import re
import math
import json
import os
import copy
parser = argparse.ArgumentParser()
# fmt: off
parser.add_argument('--model', required=True)
parser.add_argument('--datadir', default='wmt14_ende/bin')
parser.add_argument('--lmmodel', default='lm_de.arpa')
parser.add_argument('--strategy', default="beamlm200")
parser.add_argument('--alpha-range', default="1-1.4")
parser.add_argument('--outputdir', default="output")
parser.add_argument('--debug', action="store_true")

global pargs
pargs = parser.parse_args()

if pargs.debug:
    import ptvsd
    ptvsd.enable_attach()
    print("wait debug")
    ptvsd.wait_for_attach()

def runcmd(cmd, checkcode=True):
    print(cmd)
    code = os.system(cmd)
    if checkcode and os.WEXITSTATUS(code) != 0:
        raise RuntimeError(f"command exit with error {os.WEXITSTATUS(code)}")

strategies = {
    "beam200": {"decode_strategy": "beamsearch", "decode_beta": 1, "decode_beamsize": 200, "decode_top_cand_n": 5,
                "decode_gamma": 0, "decode_lm_path":None,
                "decode_max_beam_per_length":10, "decode_top_p":0.9, "decode_max_batchsize":32, "max_tokens":3096, "decode_dedup": True},
    "beam100": {"decode_strategy": "beamsearch", "decode_beta": 1, "decode_beamsize": 100, "decode_top_cand_n": 5,
                "decode_gamma": 0, "decode_lm_path":None,
                "decode_max_beam_per_length":10, "decode_top_p":0.9, "decode_max_batchsize":32, "max_tokens":3096, "decode_dedup": True},
    "beam50": {"decode_strategy": "beamsearch", "decode_beta": 1, "decode_beamsize": 50, "decode_top_cand_n": 5,
                "decode_gamma": 0, "decode_lm_path":None,
                "decode_max_beam_per_length":10, "decode_top_p":0.9, "decode_max_batchsize":32, "max_tokens":3096, "decode_dedup": True},
    "beamlm200": {"decode_strategy": "beamsearch", "decode_beta": 1, "decode_beamsize": 200, "decode_top_cand_n": 5,
                "decode_gamma": 0.1, "decode_lm_path":f"{pargs.lmmodel}",
                "decode_max_beam_per_length":10, "decode_top_p":0.9, "decode_max_batchsize":32, "max_tokens":3096, "decode_dedup": True},
    "beamlm100": {"decode_strategy": "beamsearch", "decode_beta": 1, "decode_beamsize": 100, "decode_top_cand_n": 5,
                "decode_gamma": 0.1, "decode_lm_path":f"{pargs.lmmodel}",
                "decode_max_beam_per_length":10, "decode_top_p":0.9, "decode_max_batchsize":32, "max_tokens":3096, "decode_dedup": True},
    "beamlm50": {"decode_strategy": "beamsearch", "decode_beta": 1, "decode_beamsize": 50, "decode_top_cand_n": 5,
                "decode_gamma": 0.1, "decode_lm_path":f"{pargs.lmmodel}",
                "decode_max_beam_per_length":10, "decode_top_p":0.9, "decode_max_batchsize":32, "max_tokens":3096, "decode_dedup": True},
}

def generate(sn, sargs, decode_alpha, subset):
    target_name = f"{subset}_{sn}_{decode_alpha}.txt"
    sargs['decode_alpha'] = decode_alpha

    if not os.path.exists(pargs.outputdir):
        os.makedirs(pargs.outputdir)

    if os.path.exists(f"{pargs.outputdir}/{target_name}"):
        line = open(f"{pargs.outputdir}/{target_name}").readlines()[-1]
        groups = re.search(r"BLEU4? = ([0-9.]+).*ratio\s*=\s*([0-9.]+)", line).groups()
        bleu = groups[0]
        ratio = groups[1]
        return float(bleu), float(ratio)

    other_options = ""
    if "enzh" in pargs.datadir:
        other_options += f" --source-lang en --target-lang zh --tokenizer moses --scoring sacrebleu --sacrebleu-tokenizer zh"

    sargs_str = '"' + json.dumps(sargs).replace("\"", "\\\"").replace("null", "None") + '"'

    omp_thread = 2 if subset == "test" else 8
    batch_size = 1 if subset == "test" else sargs['decode_max_batchsize']
    runcmd(f"OMP_NUM_THREADS={omp_thread} fairseq-generate \
            {pargs.datadir} \
            --gen-subset {subset} \
            --task translation_lev_modified \
            --iter-decode-max-iter 0 \
            --iter-decode-eos-penalty 0 \
            --beam 1 {other_options}\
            --batch-size {batch_size} \
            --seed 0 \
            --remove-bpe \
            --user-dir fs_plugins \
            --path {pargs.model}\
            --model-overrides {sargs_str} > {pargs.outputdir}/{target_name}")

    line = open(f"{pargs.outputdir}/{target_name}").readlines()[-1]
    groups = re.search(r"BLEU4? = ([0-9.]+).*ratio\s*=\s*([0-9.]+)", line).groups()
    bleu = groups[0]
    ratio = groups[1]

    return float(bleu), float(ratio)

def prune(decode_alpha):
    decode_alpha += 0.001
    return float(f"{decode_alpha:.2f}")

def tune_alpha(sn, sargs):
    small_step = 0.01

    left_alpha, right_alpha = pargs.alpha_range.split("-")
    left_alpha = float(left_alpha)
    right_alpha = float(right_alpha)

    left_bleu, ratio = generate(sn, sargs, prune(left_alpha), "valid")
    print(f"tune {sn}_{left_alpha}: bleu={left_bleu} ratio={ratio}")
    right_bleu, ratio = generate(sn, sargs, prune(right_alpha), "valid")
    print(f"tune {sn}_{right_alpha}: bleu={right_bleu} ratio={ratio}")
    
    while left_alpha + small_step + 1e-6 < right_alpha:
        decode_alpha1 = prune(max(math.floor((left_alpha * 2 / 3 + right_alpha / 3) / small_step + 0.5) * small_step, left_alpha + small_step))
        decode_alpha2 = prune(max(math.floor((left_alpha / 3 + right_alpha * 2 / 3) / small_step + 0.5) * small_step, left_alpha + small_step))
        bleu1, ratio1 = generate(sn, sargs, decode_alpha1, "valid")
        # print(f"tune {sn}_{decode_alpha1}: bleu={bleu1} ratio={ratio1}")
        bleu2, ratio2 = generate(sn, sargs, decode_alpha2, "valid")
        # print(f"tune {sn}_{decode_alpha2}: bleu={bleu2} ratio={ratio2}")
        
        if bleu1 < bleu2:
            left_alpha = decode_alpha1
            left_bleu = bleu1
        else:
            right_alpha = decode_alpha2
            right_bleu = bleu2

    if left_bleu >= right_bleu:
        return left_alpha
    else:
        return right_alpha

sargs = copy.copy(strategies[pargs.strategy])
alpha_optimal = tune_alpha(pargs.strategy, sargs)
bleu, ratio = generate(pargs.strategy, sargs, alpha_optimal, "test")

print(f"{pargs.strategy}_{alpha_optimal}: bleu={bleu} ratio={ratio}")

            
# %%

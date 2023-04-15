import argparse
import numpy as np
import random
import math
# import numba
from transformers import BertTokenizer

tok = BertTokenizer.from_pretrained('bert-base-uncased')

parser = argparse.ArgumentParser()
# fmt: off
parser.add_argument('file')
parser.add_argument('--max-seq-length', type=int, default=600)
parser.add_argument('--mask-ratio', type=float, default=0.15)
parser.add_argument('--mask-max-seg', type=int, default=6)
parser.add_argument('--mask-strategy', type=str, default="segment")
parser.add_argument('--mask-min-token-per-seg', type=int, default=4)
parser.add_argument('--mask-min-interval', type=int, default=8)
parser.add_argument('--duplicate', type=int, default=2)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--out', type=str)

args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)

srcfile = open(f"{args.out}.src", 'w')
tgtfile = open(f"{args.out}.tgt", 'w')

# @numba.jit(nopython=True)
def numba_process(buffer, mask_ratio, mask_max_seg, mask_min_token_per_seg, mask_min_interval):
    mask_num = int(len(buffer) * mask_ratio)

    if args.mask_strategy == "iid":
        iid_id = np.sort(np.argsort(np.random.uniform(0, 1, len(buffer)), 0)[:mask_num], 0)
        mask_length = []
        mask_interval = []
        for i, x in enumerate(iid_id):
            if i == 0:
                mask_interval.append(x)
                mask_length.append(1)
            elif iid_id[i-1] + 1 != x:
                mask_interval.append(x - iid_id[i-1] - 1)
                mask_length.append(1)
            else:
                mask_length[-1] += 1
        mask_interval.append(len(buffer) - iid_id[-1] - 1)
        mask_interval = np.array(mask_interval)
        mask_length = np.array(mask_length)
        seg_num = len(mask_length)
    else:
        max_seg = min(mask_max_seg + 1, mask_num // mask_min_token_per_seg, (len(buffer) - mask_num) // mask_min_interval + 1)
        seg_num = max_seg - 1
        mask_length = np.array([int(math.floor(i * mask_num / seg_num)) for i in range(seg_num + 1)])
        mask_length = mask_length[1:] - mask_length[:-1]

        mask_interval = np.array([0] + [mask_min_interval] * (seg_num - 1) + [0]) + \
                np.random.multinomial(len(buffer) - mask_num - mask_min_interval * (seg_num - 1), [1. / (seg_num + 1)] * (seg_num + 1))

    assert mask_length.sum() + mask_interval.sum() == len(buffer)
    assert mask_length.sum() == mask_num

    src = ["" for _ in range(0)]
    tgt = ["" for _ in range(0)]
    nowpos = 0
    for i in range(seg_num):
        nextpos = nowpos + mask_interval[i]
        src += buffer[nowpos:nextpos] + [f"[P{i}]"]
        nowpos = nextpos

        nextpos = nowpos + mask_length[i]
        tgt += [f"[P{i}]"] + buffer[nowpos:nextpos]
        nowpos = nextpos

    nextpos = nowpos + mask_interval[-1]
    src += buffer[nowpos:nextpos]
    tgt += [f"[P{seg_num}]"]

    return src, tgt


def process_and_write(buffer, srcfile, tgtfile):
    # import time
    # starttime = time.time()
    src, tgt = numba_process(buffer, args.mask_ratio, args.mask_max_seg, args.mask_min_token_per_seg, args.mask_min_interval)
    # secondtime = time.time()

    srcfile.write(" ".join(src) + "\n")
    tgtfile.write(" ".join(tgt) + "\n")

    # print(secondtime - starttime, time.time() - secondtime)

for dup in range(args.duplicate):
    buffer = []
    for i, line in enumerate(open(args.file)):
        if i % 10000 == 0:
            print(dup, i, flush=True)

        if len(buffer) > args.max_seq_length:
            buffer = []

        if not line or line.isspace():
            continue

        line = line.strip()

        # import time
        # starttime = time.time()
        if not buffer:
            nowline = tok.tokenize(" " + line)
        else:
            nowline = tok.tokenize(line)
        # print(time.time() - starttime)
        buffer += nowline

        if len(buffer) > args.max_seq_length:
            process_and_write(buffer[:args.max_seq_length], srcfile, tgtfile)
            skip_num = int((len(buffer) - args.max_seq_length) * random.random())
            buffer = buffer[args.max_seq_length + skip_num:]

srcfile.close()
tgtfile.close()

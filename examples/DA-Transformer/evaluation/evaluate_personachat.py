# pip install pycocoevalcap
# pip install nltk

from collections import Counter

import numpy as np
from argparse import ArgumentParser
from pycocoevalcap.bleu.bleu import Bleu

def distinct(seqs):
    """ Calculate intra/inter distinct 1/2. """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


def bleu(hyps, refs):
    """ Calculate bleu 1/2. """

    ref_len = 0
    hyp_len = 0
    gts = {}
    res = {}
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        ref_len += len(ref)
        hyp_len += len(hyp)
        gts[i] = [" ".join(ref)]
        res[i] = [" ".join(hyp)]

    score, scores = Bleu(4).compute_score(gts, res)
    return score[0], score[1], hyp_len / ref_len


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--golden-file', dest="golden_file", help='Input data file, one golden per line.')
    parser.add_argument('--pred-file', dest="pred_file", help='Model predictions.')
    args = parser.parse_args()

    with open(args.pred_file, encoding='utf-8') as fin:
        preds = fin.readlines()
        preds = [line.strip().split(" ") for line in preds]
    with open(args.golden_file, encoding='utf-8') as fin:
        golds = fin.readlines()
        golds = [line.strip().split(" ") for line in golds]

    bleu1, bleu2, ratio = bleu(preds, golds)
    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(preds)
    print(bleu1 * 100., bleu2 * 100., ratio, inter_dist1, inter_dist2)


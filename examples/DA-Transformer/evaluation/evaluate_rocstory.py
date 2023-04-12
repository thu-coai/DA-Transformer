# pip install pycocoevalcap
# pip install nltk

from collections import Counter

from nltk import ngrams
import numpy as np
from argparse import ArgumentParser
import string
from pycocoevalcap.bleu.bleu import Bleu

_tok_dict = {"(": "-lrb-", ")": "-rrb-",
             "[": "-lsb-", "]": "-rsb-",
             "{": "-lcb-", "}": "-rcb-",
             "[UNK]": "UNK", '&': '&amp;', '<': '&lt;', '>': '&gt;'}

def repetition_distinct(cands):
    result = {}
    for i in range(1, 5):
        num, all_ngram, all_ngram_num = 0, {}, 0.
        for k, cand in enumerate(cands):
            ngs = ["_".join(c) for c in ngrams(cand, i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
            for s in set(ngs):
                if ngs.count(s) > 1:
                    # if i >= 3:
                    #     print(s)
                    #     print(" ".join(cand))
                    #     err()
                    num += 1
                    break
        result["repetition-%d"%i] = num / float(len(cands))
        result["distinct-%d"%i] = len(all_ngram) / float(all_ngram_num)
    return result

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


def _is_digit(w):
    for ch in w:
        if not(ch.isdigit() or ch == ','):
            return False
    return True

def fix_tokenization(text):
    input_tokens = text.split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok in _tok_dict.keys():
            output_tokens.append(_tok_dict[tok])
            i += 1
        elif tok == "\"":
            if has_left_quote:
                output_tokens.append("''")
            else:
                output_tokens.append("``")
            has_left_quote = not has_left_quote
            i += 1
        elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and input_tokens[i + 1] == "t":
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
            output_tokens.append("'"+input_tokens[i + 1])
            i += 2
        elif tok == "'":
            if has_left_single_quote:
                output_tokens.append("'")
            else:
                output_tokens.append("`")
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif tok == "," and len(output_tokens) > 0 and _is_digit(output_tokens[-1]) and i < len(input_tokens) - 1 and _is_digit(input_tokens[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ','+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and input_tokens[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.'+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[-1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[i + 1].isupper() and input_tokens[i + 2] == '.':
            # U . N . -> U.N.
            k = i+3
            while k+2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i += 2
        elif tok == "-":
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == "-":
                output_tokens.append("--")
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append("-")
                i += 1
            elif output_tokens[-1] not in string.punctuation and input_tokens[i + 1][0] not in string.punctuation:
                output_tokens[-1] += "-"
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append("-")
                i += 1
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return " ".join(output_tokens)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--golden-file', dest="golden_file", help='Input data file, one golden per line.')
    parser.add_argument('--pred-file', dest="pred_file", help='Model predictions.')
    args = parser.parse_args()
    
    with open(args.pred_file, encoding='utf-8') as fin:
        preds = fin.readlines()
        preds = [fix_tokenization(line.strip()).split(" ") for line in preds]
    with open(args.golden_file, encoding='utf-8') as fin:
        golds = fin.readlines()
        golds = [line.strip().split(" ") for line in golds]
    
   
    bleu1, bleu2, ratio = bleu(preds, golds)
    result = repetition_distinct(preds)

    print(bleu1 * 100., bleu2 * 100., ratio, result['distinct-4'] * 100)


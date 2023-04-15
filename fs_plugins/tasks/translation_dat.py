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

from dataclasses import dataclass, field
from math import log
import torch
import os, itertools
from fairseq import metrics, utils
from fairseq.dataclass import ChoiceEnum
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.tasks import register_task, FairseqTask
from fairseq.data import FairseqDataset, data_utils, iterators
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from fairseq.utils import new_arange
import logging
from typing import Optional
from omegaconf import II
import numpy as np
from fairseq.dataclass import ChoiceEnum, FairseqDataclass

from .translation_dat_dict import TranslationDATDict
from .translation_dat_dataset import TranslationDATDataset

logger = logging.getLogger(__name__)

def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    truncate_source=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    upsample_scale="4~8",
    upsample_base="source",
    max_tokens_after_upsample=False,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return TranslationDATDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        eos=None,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        upsample_scale=upsample_scale,
        upsample_base=upsample_base,
        max_tokens_after_upsample=max_tokens_after_upsample,
    )


@dataclass
class TranslationDATConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_order: int = field(
        default=4, metadata={"help": "the ngram order of bleu. Default: 4"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    # options for DAT
    seg_tokens: int = field(
        default=0, metadata={"help": "This parameter specifies the number of special tokens that will be used for segment id."
                                    "If you are using pre-trained checkpoints, please set this value to 32."}
    )
    upsample_scale: str = field(
        default="4~8",
        metadata={
            "help": 'Specifies the upsample scale for the decoder input length in training. For instance, "4~8" indicates that the upsampling scale will be uniformly sampled from the range [4, 8];'
                    '"4" indicates fixed upsampling scale.'
        }
    )
    upsample_base: str = field(
        default="source", metadata={"help": 'Possible values are: ["predict", "source", "source_old"]. '
            'If set to "predict", the DAG size will be determined by the golden target length during training and the predicted length during inference. Note that --length-loss-factor must be greater than 0 during training. '
            'If set to "source", the DAG size will be determined by the source length during both training and inference. You can disable the length predictor during training by setting --length-loss-factor to 0. '
            'If set to "source_old", the DAG size length is determined similarly to "source" but several token longer. This option is only used for compatibility with the upsampling method in version 1.0.'}
    )
    max_tokens_after_upsample: bool = field(
        default=False, metadata={"help": "If enabled, the maximum number of tokens (--max-tokens) considered during generation "
            "will take into account the upsampling ratio. In other words, the length of the generated sequence will be capped at "
            "max(source_length, decoder_length * upsample_scale). Default: False."}
    )
    filter_ratio: Optional[str] = field(
        default=None, metadata={"help": "Filters out samples that do not satisfy the specified len(target)/len(source) ratio constraints. "
            'For example, if the ratio is set to "8", samples where len(target)/len(source) > 8 or len(target)/len(source) < 1/8 will be removed. '
            'If set to "0.5~2", samples where len(target)/len(source) < 0.5 or len(target)/len(source) > 2 will be removed. '
            "Default: None (disabled)."}
    )
    prepend_bos: bool = field(
        default=False, metadata={"help": "Prepends the beginning of sentence token (bos) to the source sequence. "
            "(Note: The target sequence always contains both bos and eos tokens in the DA-Transformer.) Default: False."}
    )

    decode_upsample_scale: Optional[float] = field(
        default=None, metadata={"help": "Upsampling scale to determine the DAG size during inference."}
    )
    decode_strategy: str = field(
        default="lookahead",
        metadata={"help": 'Decoding strategy to use. Options include "greedy", "lookahead", "viterbi", "jointviterbi", "sample", and "beamsearch".'}
    )

    decode_no_consecutive_repeated_ngram: int = field(
        default=0, metadata={
            "help": "Prevent consecutive repeated k-grams (k <= n) in the generated text. Use 0 to disable this feature. This argument is used in greedy, lookahead, sample, and beam search decoding methods."
        }
    )
    decode_no_repeated_ngram: int = field(
        default=0, metadata={
            "help": "Prevent repeated k-grams (not necessarily consecutive) with order n or higher in the generated text. Use 0 to disable this feature. This argument is used in greedy, lookahead, sample, and beam search decoding methods."
        }
    )
    decode_top_cand_n: float = field(
        default=5, metadata={
            "help": "Number of top candidates to consider during transition. This argument is used in greedy and lookahead decoding with ngram prevention, and sample and beamsearch decoding methods."
        }
    )
    decode_top_p: float = field(
        default=0.9, metadata={
            "help": "Maximum probability of top candidates to consider during transition. This argument is used in greedy and lookahead decoding with ngram prevention, and sample and beamsearch decoding methods."
        }
    )
    decode_viterbibeta: float = field(
        default=1, metadata={
            "help": "Parameter used for length penalty in Viterbi decoding. The sentence with the highest score is found using: P(A,Y|X) / |Y|^{beta}"
        }
    )
    decode_temperature: float = field(
        default=1, metadata={
            "help": "Temperature to use in sample decoding."
        }
    )

    decode_beamsize: float = field(
        default=100, metadata={"help": "Beam size used in beamsearch decoding."}
    )
    decode_max_beam_per_length: float = field(
            default=10, metadata={"help": "Maximum number of beams with the same length in each step during beamsearch decoding."}
        )
    decode_gamma: float = field(
            default=0.1, metadata={"help": "Parameter used for n-gram language model score in beamsearch decoding. "
                                            "The sentence with the highest score is found using: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]"}
        )
    decode_alpha: float = field(
            default=1.1, metadata={"help": "Parameter used for length penalty in beamsearch decoding. "
                                            "The sentence with the highest score is found using: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]"}
        )
    decode_beta: float = field(
            default=1, metadata={"help": "Parameter used to scale the score of logits in beamsearch decoding. "
                                            "The score of a sentence is given by: sum P(y_i|a_i) + beta * sum log(a_i|a_{i-1})"}
        )
    decode_lm_path: Optional[str] = field(
            default=None, metadata={"help": "Path to n-gram language model to use during beamsearch decoding. Set to None to disable n-gram LM."}
        )
    decode_max_batchsize: int = field(
            default=32, metadata={"help": "Maximum batch size to use during beamsearch decoding. "
                                            "Should not be smaller than the actual batch size, as it is used for memory allocation."}
        )
    decode_max_workers: int = field(
            default=1, metadata={"help": 'Number of multiprocess workers to use during beamsearch decoding. '
                                        'More workers will consume more memory. It does not affect decoding latency but decoding throughtput, '
                                        'so you must use "fariseq-fastgenerate" to enable the overlapped decoding to tell the difference.'}
        )
    decode_threads_per_worker: int = field(
            default=4, metadata={"help": "Number of threads per worker to use during beamsearch decoding. "
                                            "This setting also applies to both vanilla decoding and overlapped decoding. A value between 2 and 8 is typically optimal."}
        )
    decode_dedup: bool = field(
        default=False, metadata={"help": "Enable token deduplication in BeamSearch."}
    )
    max_encoder_batch_tokens: Optional[int] = field(
        default=None,
        metadata={"help": 'Specifies the maximum number of tokens for the encoder input to avoid running out of memory. The default value of None indicates no limit.'},
    )
    max_decoder_batch_tokens: Optional[int] = field(
        default=None,
        metadata={"help": 'Specifies the maximum number of tokens for the decoder input to avoid running out of memory. The default value of None indicates no limit.'},
    )
    do_not_load_task_args: bool = field(
        default=False, metadata={"help": "Do not load task arguments stored in checkpoints."}
    )

@register_task("translation_dat_task", dataclass=TranslationDATConfig)
class TranslationDATTask(TranslationTask):

    cfg: TranslationDATConfig

    def __init__(self, cfg):
        FairseqTask.__init__(self, cfg)

    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):
        return cls(cfg)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=self.cfg.prepend_bos,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            upsample_scale=self.cfg.upsample_scale,
            upsample_base=self.cfg.upsample_base,
            max_tokens_after_upsample=self.cfg.max_tokens_after_upsample,
        )

    @classmethod
    def load_dictionary(cls, filename, seg_tokens):
        return TranslationDATDict.load(filename, seg_tokens=seg_tokens)

    def build_generator(self, models, args, **unused):
        from .translation_dat_generator import TranslationDATGenerator
        return TranslationDATGenerator(self.target_dictionary)

    def build_model(self, cfg, from_checkpoint=False):
        # find explicit task arguments
        from fairseq.options import get_parser, add_dataset_args
        from fairseq import options
        parser = get_parser("Task", "translation_dat_task")
        add_dataset_args(parser)
        args, _ = options.parse_args_and_arch(parser, suppress_defaults=True, parse_known=True)

        # update arguments for decoding
        for key, value in vars(args).items():
            if hasattr(cfg, key) and getattr(cfg, key) != value:
                logging.info(f"Updating model arguments by user commands, key={key}, value_in_checkpoint={getattr(cfg, key)} -> value_in_cmd={value}")
                setattr(cfg, key, value)
        # setting fp16
        if not getattr(args, "fp16", False) and hasattr(cfg, "fp16") and getattr(cfg, "fp16"):
            logging.info(f"Updating model arguments by user commands, key=fp16, value_in_checkpoint=True -> value_in_cmd=False")
            setattr(cfg, key, value)

        # update task cfg by model checkpoints
        if getattr(args, "do_not_load_task_args", None):
            logging.info("You use --do-not-load-task-args, then you have to specify all task arguments explicitly.")
        else:
            for key, value in vars(cfg).items():
                if key not in {"_name"} and hasattr(self.cfg, key) and getattr(self.cfg, key) != value:
                    # logging.info(f"Updating task arguments by model arguments, key={key}, oldvalue={getattr(self.cfg, key)}, newvalue={value}")
                    setattr(self.cfg, key, value)

        # build dictionary
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if self.cfg.source_lang is None or self.cfg.target_lang is None:
            self.cfg.source_lang, self.cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if self.cfg.source_lang is None or self.cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )
        self.src_dict = TranslationDATDict.load(
            os.path.join(paths[0], "dict.{}.txt".format(self.cfg.source_lang)), seg_tokens=self.cfg.seg_tokens
        )
        self.tgt_dict = TranslationDATDict.load(
            os.path.join(paths[0], "dict.{}.txt".format(self.cfg.target_lang)), seg_tokens=self.cfg.seg_tokens
        )
        assert self.cfg.source_lang is not None and self.cfg.target_lang is not None
        assert self.src_dict.pad() == self.tgt_dict.pad()
        assert self.src_dict.eos() == self.tgt_dict.eos()
        assert self.src_dict.unk() == self.tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(self.cfg.source_lang, len(self.src_dict)))
        logger.info("[{}] dictionary: {} types".format(self.cfg.target_lang, len(self.tgt_dict)))

        model = super().build_model(cfg, from_checkpoint)
        return model

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_dat_task task is not supported"
            )

        return TranslationDATDataset(
            src_tokens, src_lengths, self.source_dictionary,
            left_pad_source=self.cfg.left_pad_source,
            eos=None,
            prepend_bos=self.cfg.prepend_bos,
            num_buckets=self.cfg.num_buckets,
            upsample_scale=self.cfg.upsample_scale,
            upsample_base=self.cfg.upsample_base,
            max_tokens_after_upsample=self.cfg.max_tokens_after_upsample,
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        # print(update_num)
        sample['update_num'] = update_num
        if ignore_grad:
            sample['dummy'] = True
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)

            EVAL_BLEU_ORDER = self.cfg.eval_bleu_order
            import sacrebleu
            if sacrebleu.BLEU.NGRAM_ORDER != self.cfg.eval_bleu_order:
                sacrebleu.BLEU.NGRAM_ORDER = self.cfg.eval_bleu_order
                func = sacrebleu.BLEU.extract_ngrams
                sacrebleu.BLEU.extract_ngrams = lambda x: func(x, min_order=1, max_order=self.cfg.eval_bleu_order)
            if self.cfg.eval_bleu:
                bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
                logging_output["_bleu_sys_len"] = bleu.sys_len
                logging_output["_bleu_ref_len"] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                    logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output


    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            if self.cfg.target_lang == "ja":
                return sacrebleu.corpus_bleu(hyps, [refs], tokenize="ja-mecab")
            elif self.cfg.target_lang == "zh":
                return sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
            else:
                return sacrebleu.corpus_bleu(hyps, [refs])

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Modified for DAT training: support filter samples by tar/src length ratios

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = (
            not disable_iterator_cache
            and not update_epoch_batch_itr
            and self.can_reuse_epoch_itr(dataset)
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None or self.filter_ratio is not None:
            indices = self.filter_indices_by_size_and_ratio(
                indices, dataset, max_positions, ignore_invalid_inputs, self.cfg.filter_ratio
            )

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
            grouped_shuffling=grouped_shuffling,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def filter_indices_by_size_and_ratio(
        self, indices, dataset, max_positions=None, ignore_invalid_inputs=False, filter_ratio=None
    ):
        """
        Filter examples that are too large

        Args:
            indices (np.array): original array of sample indices
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
        Returns:
            np.array: array of filtered sample indices
        """
        def filter_indices_by_size_and_ratio(indices, max_sizes, filter_ratio):
            """Filter a list of sample indices. Remove those that are longer
                than specified in max_sizes.

            Args:
                indices (np.array): original array of sample indices
                max_sizes (int or list[int] or tuple[int]): max sample size,
                    can be defined separately for src and tgt (then list or tuple)

            Returns:
                np.array: filtered sample array
                list: list of removed indices
            """
            src_sizes = dataset.src_sizes
            tgt_sizes = dataset.tgt_sizes
            if max_sizes is None and filter_ratio is None:
                return indices, []
            if type(max_sizes) in (int, float):
                max_src_size, max_tgt_size = max_sizes, max_sizes
            else:
                max_src_size, max_tgt_size = max_sizes

            if filter_ratio is not None:
                if "~" in filter_ratio:
                    filter_ratio_rev, filter_ratio_max = filter_ratio.split("~")
                    filter_ratio_rev, filter_ratio_max = float(filter_ratio_rev), float(filter_ratio_max)
                    if filter_ratio_rev == 0:
                        filter_ratio_rev = 100000
                    else:
                        filter_ratio_rev = 1 / filter_ratio_rev
                else:
                    filter_ratio_rev, filter_ratio_max = float(filter_ratio), float(filter_ratio)

            if tgt_sizes is None:
                ignored = indices[src_sizes[indices] > max_src_size]
            else:
                if filter_ratio is not None:
                    ignored = indices[
                        (src_sizes[indices] > max_src_size) | (tgt_sizes[indices] > max_tgt_size) |
                        ((src_sizes[indices] * filter_ratio_max).astype(int) < tgt_sizes[indices]) |
                        ((tgt_sizes[indices] * filter_ratio_rev).astype(int) < src_sizes[indices])
                    ]
                else:
                    ignored = indices[
                        (src_sizes[indices] > max_src_size) | (tgt_sizes[indices] > max_tgt_size)
                    ]

            if len(ignored) > 0:
                if tgt_sizes is None:
                    indices = indices[src_sizes[indices] <= max_src_size]
                else:
                    if filter_ratio is not None:
                        indices = indices[
                            (src_sizes[indices] <= max_src_size)
                            & (tgt_sizes[indices] <= max_tgt_size)
                            & ((src_sizes[indices] * filter_ratio_max).astype(int) >= tgt_sizes[indices])
                            & ((tgt_sizes[indices] * filter_ratio_rev).astype(int) >= src_sizes[indices])
                        ]
                    else:
                        indices = indices[
                            (src_sizes[indices] <= max_src_size)
                            & (tgt_sizes[indices] <= max_tgt_size)
                        ]
            return indices, ignored.tolist()

        indices, ignored = filter_indices_by_size_and_ratio(indices, max_positions, filter_ratio if ignore_invalid_inputs else None)
        if len(ignored) > 0:
            if not ignore_invalid_inputs:
                raise Exception(
                    (
                        "Size of sample #{} is invalid (={}) since max_positions={} and filter_ratio={}, "
                        "skip this example with --skip-invalid-size-inputs-valid-test"
                    ).format(ignored[0], dataset.size(ignored[0]), max_positions, filter_ratio)
                )
            logger.warning(
                (
                    "{:,} samples have invalid sizes and will be skipped, "
                    "max_positions={} and filter_ratio={}, first few sample ids={}"
                ).format(len(ignored), max_positions, filter_ratio, ignored[:10])
            )
        return indices

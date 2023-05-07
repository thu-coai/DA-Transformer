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

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils

logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None, pad_idx=pad_idx):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        prev_output_tokens_segid = merge("prev_output_tokens_segid", left_pad=left_pad_target, pad_idx=0)
        prev_output_tokens_segid = prev_output_tokens_segid.index_select(0, sort_order)
        if samples[0]['force_emit'] is not None:
            force_emit = merge("force_emit", left_pad=left_pad_target, pad_idx=-1)
            force_emit = force_emit.index_select(0, sort_order)
            bound_end = merge("bound_end", left_pad=left_pad_target, pad_idx=0)
            bound_end = bound_end.index_select(0, sort_order)
            tgt_segid = merge("tgt_segid", left_pad=left_pad_target, pad_idx=-1)
            tgt_segid = tgt_segid.index_select(0, sort_order)
        else:
            force_emit = None
            bound_end = None
            tgt_segid = None
    else:
        prev_output_tokens = None
        prev_output_tokens_segid = None
        force_emit = None
        bound_end = None
        tgt_segid = None
        target = None
        ntokens = None

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "prev_output_tokens": prev_output_tokens,
            "prev_output_tokens_segid": prev_output_tokens_segid,
            "force_emit": force_emit,
            "bound_end": bound_end,
            "tgt_segid": tgt_segid
        },
        "target": target,
    }

    return batch


class TranslationDATDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets modified for Directed Acyclic Transformer.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
        upsample_scale (str, optional): the upsample scale for decoder input length.
            E.g.: 4~8 for uniformly sampling from [4, 8].
        upsample_base (str, optional): Either ``predict'' or ``source''. If ``predict'',
            the decoder input is determined by golden target length in training, and by
            predicted length in inference. Note --length-loss-factor should > 0. If
            ``source'', the decoder input is determined by the source length in training
            and inference. You can set --length-loss-factor to 0 to disable length predictor.
        max_tokens_after_upsample (bool, optional): if True, --max-tokens consider upsampling
            ratio in calculating the token number. Specifially, the sample length is considered
            as max(source_length, decoder_length * upsample_scale). Default: False.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
        # for DAT:
        upsample_scale="4~8",
        upsample_base="source",
        max_tokens_after_upsample=False,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id

        if "~" in upsample_scale:
            self.upsample_scale_min, self.upsample_scale_max = (float(x) for x in upsample_scale.split("~"))
        else:
            self.upsample_scale_min, self.upsample_scale_max = float(upsample_scale), float(upsample_scale)
        self.upsample_base = upsample_base
        assert upsample_base in ["source", "source_old", "predict", "fixed"]
        self.max_tokens_after_upsample = max_tokens_after_upsample
        assert not max_tokens_after_upsample or upsample_base in ["predict", "fixed"], "--max-tokens-after-upsample should be used with upsample_base = predict or fixed."

        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.compat.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        if tgt_item is None:
            example = {
                "id": index,
                "source": src_item,
                "target": tgt_item,
                "prev_output_tokens": None, # [upsample]
                "prev_output_tokens_segid": None, # [upsample]
                "force_emit": None, # [upsample]
                "bound_end": None, # [upsample]
                "tgt_segid": None, #[tgt_length]
            }
            return example

        if self.upsample_base in ["source", "source_old", "fixed"]:
            # Upsampling based on the source length, usually for translation

            # make sure tgt with bos and eos
            concat_list = []
            if tgt_item[0] != self.tgt_dict.bos_index:
                concat_list.append(torch.zeros(1, dtype=torch.long).fill_(self.tgt_dict.bos_index))
            concat_list.append(tgt_item)
            if tgt_item[-1] != self.tgt_dict.eos_index:
                concat_list.append(torch.zeros(1, dtype=torch.long).fill_(self.tgt_dict.eos_index))
            if len(concat_list) > 0:
                tgt_item = torch.cat(concat_list, 0)

            if self.upsample_base == "source":
                length_src = torch.sum(src_item.ne(self.src_dict.pad_index), -1)
                length_src_special = torch.sum(src_item.eq(self.src_dict.bos_index) | src_item.eq(self.src_dict.eos_index), -1)
                upsample_factor = np.random.rand() * (self.upsample_scale_max - self.upsample_scale_min) + self.upsample_scale_min
                upsample_len = (length_src - length_src_special) * upsample_factor + length_src_special
            elif self.upsample_base == "source_old":
                upsample_factor = np.random.rand() * (self.upsample_scale_max - self.upsample_scale_min) + self.upsample_scale_min
                upsample_len = len(src_item) * upsample_factor # compatitable for older version
            elif self.upsample_base == "fixed":
                upsample_len = np.random.rand() * (self.upsample_scale_max - self.upsample_scale_min) + self.upsample_scale_min + 2
            else:
                raise NotImplementedError(f"Unknown upsample_base: {self.upsample_base}")

            upsample_len = int(upsample_len)
            prev_output_tokens = torch.zeros(upsample_len, dtype=torch.long).fill_(self.tgt_dict.unk_index)
            prev_output_tokens[0] = self.tgt_dict.bos_index
            prev_output_tokens[-1] = self.tgt_dict.eos_index
            prev_output_tokens_segid = torch.ones_like(prev_output_tokens)
            prev_output_tokens_segid[-1] = 2

            force_emit = None
            bound_end = None
            tgt_segid = None

        elif self.upsample_base == "predict":
            # Upsampling based on the golden target length, for pre-training and fine-tuning
            # The following processing can be used for both pre-training and fine-tuning.
            # For pre-training, the target should contain plan tokens, so the following codes
            #     will do the up-sampling for each consecutive mask segments independently.

            if not self.tgt_dict.is_seg_token(tgt_item[0]):
                # not a pretrain dataset; add go and eos
                concat_list = []
                if tgt_item[0] != self.tgt_dict.bos_index:
                    concat_list.append(torch.zeros(1, dtype=torch.long).fill_(self.tgt_dict.bos_index))
                concat_list.append(tgt_item)
                if tgt_item[-1] != self.tgt_dict.eos_index:
                    concat_list.append(torch.zeros(1, dtype=torch.long).fill_(self.tgt_dict.eos_index))
                if len(concat_list) > 0:
                    tgt_item = torch.cat(concat_list, 0)
            else:
                # remove eos if there is a plan token at the end
                if self.tgt_dict.is_special_token(tgt_item[-1]) and self.tgt_dict.is_seg_token(tgt_item[-2]):
                    tgt_item = tgt_item[:-1]

            bound_pos = [] # index of plan token in original target
            upsample_bound_pos = [0] # index of plan token in decoder input
            tgt_segid = torch.zeros(len(tgt_item), dtype=torch.long).fill_(-1) # the segment id for each token in target
            for i, tok in enumerate(tgt_item.numpy()):
                if self.tgt_dict.is_special_token(tok):
                    bound_pos.append(i)
                    if len(bound_pos) > 1:
                        seg_len = i - bound_pos[-2] - 1
                        upsample_factor = np.random.rand() * (self.upsample_scale_max - self.upsample_scale_min) + self.upsample_scale_min
                        upsample_bound_pos.append(upsample_bound_pos[-1] + int(seg_len * upsample_factor) + 1)
                tgt_segid[i] = len(bound_pos)
            upsample_len = upsample_bound_pos[-1] + 1

            prev_output_tokens = torch.zeros(upsample_len, dtype=torch.long).fill_(self.tgt_dict.unk_index) # decoder input
            prev_output_tokens_segid = torch.zeros(upsample_len, dtype=torch.long) # the segment id for each token in decoder input
            force_emit = -torch.ones(upsample_len, dtype=torch.long) # forced aligned poisition for each token in decoder input
            bound_end = torch.zeros(upsample_len, dtype=torch.long) # segment end position for each token in decoder input

            for ui, oi in zip(upsample_bound_pos, bound_pos):
                prev_output_tokens[ui] = tgt_item[oi]
                force_emit[ui] = oi

            for i in range(1, len(upsample_bound_pos)):
                prev_output_tokens_segid[upsample_bound_pos[i-1]:upsample_bound_pos[i]] = i
                bound_end[upsample_bound_pos[i-1]:upsample_bound_pos[i]] = upsample_bound_pos[i]
            prev_output_tokens_segid[-1] = len(upsample_bound_pos)
            bound_end[-1] = upsample_len - 1

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            "prev_output_tokens": prev_output_tokens, # [upsample]
            "prev_output_tokens_segid": prev_output_tokens_segid, # [upsample]
            "force_emit": force_emit, # [upsample]
            "bound_end": bound_end, # [upsample]
            "tgt_segid": tgt_segid, #[tgt_length]
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        if self.max_tokens_after_upsample:
            if self.upsample_base == "predict":
                return max(
                    self.src_sizes[index],
                    (self.tgt_sizes[index] * self.upsample_scale_max)  if self.tgt_sizes is not None else 0,
                )
            elif self.upsample_base == "fixed":
                return max(
                    self.src_sizes[index],
                    self.upsample_scale_max  if self.tgt_sizes is not None else 0,
                )
            else:
                raise NotImplementedError("Unknown upsample_base")
        else:
            return max(
                self.src_sizes[index],
                self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
            )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            if self.max_tokens_after_upsample:
                if self.upsample_base == "predict":
                    sizes = np.maximum(sizes, self.tgt_sizes[indices] * self.upsample_scale_max)
                elif self.upsample_base == "fixed":
                    sizes = np.maximum(sizes, self.tgt_sizes[indices] * 0 + self.upsample_scale_max)
                else:
                    raise NotImplementedError("Unknown upsample_base")
            else:
                sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
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
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )

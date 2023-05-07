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

# This file is modified from https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/bart/hub_interface.py


import logging
from typing import Dict, List
import copy

import numpy as np
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.hub_utils import GeneratorHubInterface


logger = logging.getLogger(__name__)


class DATHubInterface(GeneratorHubInterface):

    def __init__(self, cfg, task, model):
        super().__init__(cfg, task, [model])
        self.model = self.models[0]

    def encode(
        self, sentence: str, *addl_sentences, no_separator=True
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).
        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).
        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`
        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::
            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        if self.tokenizer:
            sentence = self.tokenizer.encode(sentence)
        tokens = self.bpe.encode(sentence)

        max_position = self.max_positions[0] - 1
        if self.task.cfg.prepend_bos:
            max_position -= 1

        if len(tokens.split(" ")) > max_position:
            if self.task.cfg.truncate_source:
                tokens = " ".join(tokens.split(" ")[:max_position])
            else:
                raise RuntimeError(f"Input is too long. Current input length: {len(tokens.split(' '))}. Supported max length: {max_position}.")

        if self.task.cfg.prepend_bos:
            tokens = "<s> " + tokens
        bpe_sentence = tokens + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.target_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.task.target_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.bpe.decode(self.task.target_dictionary.string(s)) for s in sentences
        ]
        if self.tokenizer:
            sentences = [
                self.tokenizer.decode(s) for s in sentences
            ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_sample(self, src_tokens: List[torch.LongTensor]):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(lambda tensor: tensor.to(self.device), sample)
        return sample

    def generate_graph(self, sentence: str):
        tokenized_sentence = self.encode(sentence)
        gen_args = copy.deepcopy(self.cfg.generation)
        generator = self.task.build_generator(
            self.models,
            gen_args,
        )
        tokenized_sentences = [tokenized_sentence]
        for batch in self._build_batches(tokenized_sentences, False): # only one batch
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            with torch.no_grad():
                decoder_out, graph_info =  generator.generate_graph(
                    self.models, batch
                )
            break
        return self.decode(decoder_out.output_tokens[0]), graph_info

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        *args,
        inference_step_args=None,
        skip_invalid_size_inputs=False,
        **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:
        inference_step_args = inference_step_args or {}
        if "prefix_tokens" in inference_step_args:
            raise NotImplementedError("prefix generation not implemented for DA-Transformer")
        res = []
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
            src_tokens = batch["net_input"]["src_tokens"]
            results = super().generate(
                src_tokens,
                *args,
                inference_step_args=inference_step_args,
                skip_invalid_size_inputs=skip_invalid_size_inputs,
                **kwargs
            )
            for id, hypos in zip(batch["id"].tolist(), results):
                res.append((id, hypos))
        res = [hypos for _, hypos in sorted(res, key=lambda x: x[0])]
        return res

    def extract_features(
        self, tokens: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError(
                "tokens exceeds maximum length: {} > {}".format(
                    tokens.size(-1), self.model.max_positions()
                )
            )
        tokens.to(device=self.device),
        prev_output_tokens = tokens.clone()

        prev_output_tokens[:, 0] = tokens.gather(
            1,
            (tokens.ne(self.task.source_dictionary.pad()).sum(dim=1) - 1).unsqueeze(-1),
        ).squeeze()

        prev_output_tokens[:, 1:] = tokens[:, :-1]
        features, extra = self.model(
            src_tokens=tokens,
            src_lengths=None,
            prev_output_tokens=prev_output_tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        # self.model.register_classification_head(
        #     name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        # )
        raise NotImplementedError("DA-Transformer does not support classfication")

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        # if tokens.dim() == 1:
        #     tokens = tokens.unsqueeze(0)
        # features = self.extract_features(tokens.to(device=self.device))
        # sentence_representation = features[
        #     tokens.eq(self.task.source_dictionary.eos()), :
        # ].view(features.size(0), -1, features.size(-1))[:, -1, :]

        # logits = self.model.classification_heads[head](sentence_representation)
        # if return_logits:
        #     return logits
        # return F.log_softmax(logits, dim=-1)
        raise NotImplementedError("DA-Transformer does not support classfication")

    def fill_mask(
        self,
        masked_inputs: List[str],
        **generate_kwargs
    ):
        masked_token = "<mask>"
        batch_tokens = []
        for masked_input in masked_inputs:
            assert (
                masked_token in masked_input
            ), "please add one {} token for the input".format(masked_token)

            text_spans = masked_input.split(masked_token)
            text_spans_bpe = []
            for i, text_span in enumerate(text_spans):
                text_spans_bpe.append(f"[P{i}]")
                text_spans_bpe.append(self.bpe.encode(text_span.rstrip()))
            text_spans_bpe = " ".join(text_spans_bpe)
            # (
            #     (" {0} ".format(masked_token))
            #     .join([self.bpe.encode(text_span.rstrip()) for text_span in text_spans])
            #     .strip()
            # )
            tokens = self.task.source_dictionary.encode_line(
                "<s> " + text_spans_bpe + " </s>",
                append_eos=False,
                add_if_not_exist=False,
            ).long()
            batch_tokens.append(tokens)

        batch_hypos = self.generate(batch_tokens, **generate_kwargs)

        return [
            [(self.decode(hypo["tokens"]), hypo["score"]) for hypo in hypos]
            for hypos in batch_hypos
        ]

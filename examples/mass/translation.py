#from fairseq.data import BertDictionary

from fairseq.tasks import register_task
from fairseq import metrics, utils
from fairseq.tasks.translation import TranslationTask, TranslationConfig

from .bert_dictionary import BertDictionary
import torch
import logging

logger = logging.getLogger(__name__)

@register_task('translation_mass', dataclass=TranslationConfig)
class TranslationMASSTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    def valid_step(self, sample, model, criterion, ema_model=None):
        if ema_model is not None:
            model = ema_model
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

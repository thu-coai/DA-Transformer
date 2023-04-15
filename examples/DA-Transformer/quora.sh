data_dir=/path/to/binarized/data/dir
checkpoint_dir=/path/to/checkpoint/dir
tensorboard_dir=/path/to/tensorboard/dir
pretrained_model=/path/to/model.bin
log_txt=/path/to/logfile

CUDA_VISIBLE_DEVICES=0,1 fairseq-train ${data_dir}  \
    \
    `# loading DA-Transformer plugins` \
    --user-dir fs_plugins \
    \
    `# DA-Transformer Task Configs` \
    --task translation_dat_task \
    --upsample-base predict --upsample-scale 4~8 \
    --seg-tokens 32 --filter-max-length 512:128 \
    \
    `# DA-Transformer Architecture Configs` \
    --arch glat_decomposed_link_pretrain \
    --links-feature feature:position --segment-embedding \
    --max-source-positions 512 --max-target-positions 1024 --truncate-source \
    --encoder-learned-pos --decoder-learned-pos \
    --share-all-embeddings --activation-fn gelu --apply-bert-init \
    --load-pretrained-model ${pretrained_model} \
    \
    `# DA-Transformer Decoding Configs (See more in the decoding section)` \
    --decode-strategy lookahead --decode-upsample-scale 6.0 \
    \
    `# DA-Transformer Criterion Configs` \
    --criterion nat_dag_loss \
    --length-loss-factor 0.1 --max-transition-length 99999 \
    --glat-p 0.3 --glance-strategy number-random \
    --no-force-emit \
    \
    `# Optimizer & Regularizer Configs` \
    --optimizer adam --adam-betas '(0.9,0.999)' --fp16 \
    --label-smoothing 0.0 --weight-decay 0.01 --dropout 0.3 \
    --lr-scheduler inverse_sqrt  --warmup-updates 10000   \
    --clip-norm 0.1 --lr 1e-5 --warmup-init-lr '1e-07' --stop-min-lr '1e-09' \
    \
    `# Training Configs` \
    `# these args leads to about 8k target tokens in a batch` \
    --max-tokens 4500  --max-tokens-valid 2048 --update-freq 1 \
    --max-encoder-batch-tokens 20000 --max-decoder-batch-tokens 20000 \
    --max-update 60000  --grouped-shuffling \
    --seed 0 --ddp-backend c10d --required-batch-size-multiple 1 \
    \
    `# Validation Configs` \
    --valid-subset valid \
    --validate-interval 9999999 --validate-interval-updates 2000 \
    --eval-bleu --eval-bleu-detok space --eval-bleu-remove-bpe bert --eval-bleu-print-samples --eval-bleu-order 4 \
    --fixed-validation-seed 7 \
    \
    `# Checkpoint Configs` \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-interval 9999999 --save-interval-updates 2000 \
    --keep-best-checkpoints 5 --save-dir ${checkpoint_dir} \
    \
    `# Logging Configs` \
    --tensorboard-logdir ${tensorboard_dir} \
    --log-format 'simple' --log-interval 100 2> >(tee -a ${log_txt}) | tee -a ${log_txt}
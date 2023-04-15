data_dir=/path/to/binarized/data/dir
checkpoint_dir=/path/to/checkpoint/dir
tensorboard_dir=/path/to/tensorboard/dir
log_txt=/path/to/logfile

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train ${data_dir}  \
    \
    `# loading DA-Transformer plugins` \
    --user-dir fs_plugins \
    \
    `# DA-Transformer Task Configs` \
    --task translation_dat_task \
    --upsample-base source --upsample-scale 8 \
    --filter-max-length 128:1024 --filter-ratio 2 \
    \
    `# DA-Transformer Architecture Configs` \
    --arch ls_glat_decomposed_link_base \
    --links-feature feature:position \
    --max-source-positions 128 --max-target-positions 1024 \
    --encoder-learned-pos --decoder-learned-pos \
    --activation-fn gelu --apply-bert-init \
    \
    `# DA-Transformer Decoding Configs (See more in the decoding section)` \
    --decode-strategy lookahead --decode-upsample-scale 8.0 \
    \
    `# DA-Transformer Criterion Configs` \
    --criterion nat_dag_loss \
    --length-loss-factor 0 --max-transition-length 99999 \
    --glat-p 0.5:0.1@200k --glance-strategy number-random \
    --no-force-emit \
    \
    `# Optimizer & Regularizer Configs` \
    --optimizer ls_adam --adam-betas '(0.9,0.999)' --fp16 \
    --label-smoothing 0.0 --weight-decay 0.01 --dropout 0.1 \
    --lr-scheduler inverse_sqrt  --warmup-updates 10000   \
    --clip-norm 0.1 --lr 0.0005 --warmup-init-lr '1e-07' --stop-min-lr '1e-09' \
    \
    `# Training Configs` \
    --max-tokens 4096  --max-tokens-valid 4096 --update-freq 2 \
    --max-update 300000  --grouped-shuffling \
    --max-encoder-batch-tokens 8000 --max-decoder-batch-tokens 34000 \
    --seed 0 --ddp-backend c10d --required-batch-size-multiple 1 \
    \
    `# Validation Configs` \
    --valid-subset valid \
    --validate-interval 1       --validate-interval-updates 10000 \
    --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --fixed-validation-seed 7 \
    \
    `# Checkpoint Configs` \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-interval 1  --save-interval-updates 10000 \
    --keep-best-checkpoints 5 --save-dir ${checkpoint_dir} \
    \
    `# Logging Configs` \
    --tensorboard-logdir ${tensorboard_dir} \
    --log-format 'simple' --log-interval 100 2> >(tee -a ${log_txt}) | tee -a ${log_txt}
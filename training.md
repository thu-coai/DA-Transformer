# Training Configs

### Task Configs

```bash
--task translation_dat_task     # Task for DA-Transformer
--upsample-base predict         # Possible values are: ["predict", "source", "source_old"].
                                #   If set to "predict", the DAG size will be determined by the golden target length during training and the predicted length during inference.
                                #   Note that --length-loss-factor must be greater than 0 during training.
                                #   If set to "source", the DAG size will be determined by the source length during both training and inference. You can disable the length
                                #   predictor during training by setting --length-loss-factor to 0.
                                #   If set to "source_old", the DAG size is determined similarly to "source" but several token longer. This option is only used for
                                #   compatibility with the upsampling method in version 1.0.
--upsample-scale 4~8            # Specifies the upsample scale for the decoder input length in training.
                                #   For instance, "4~8" indicates that the upsampling scale will be uniformly sampled from the range [4, 8];
                                #   "4" indicates fixed upsampling scale.
--seg-tokens 32                 # This parameter specifies the number of special tokens that will be used for segment id.
                                #   If you are using pre-trained checkpoints, please set this value to 32.
--filter-max-length 512:128     # Filters samples that exceed the maximum lengths. For example, "128:256" indicates a maximum source length of 128 and a maximum target length of 256.
                                #   The default value of None filters according to max-source-positions and max-target-positions.
--filter-ratio 8                # Filters out samples that do not satisfy the specified len(target)/len(source) ratio constraints.
                                #   For example, if the ratio is set to "8", samples where len(target)/len(source) > 8 or len(target)/len(source) < 1/8 will be removed.
                                #   If set to "0.5~2", samples where len(target)/len(source) < 0.5 or len(target)/len(source) > 2 will be removed.
                                #   Default: None (disabled).
```

### Model Configs

```bash
--arch glat_decomposed_link_base    # The Model Architecture. You can use "ls_glat_decomposed_link_base" to enable LightSeq's Transformer
--links-feature feature:position    # Specifies the features used to predict transitions, separated by a colon.
                                    # For example, "feature:position" represents the concatenation of decoder features and learnable positional embeddings.
--segment-embedding                 # Adds an additional embedding represented segment id for the decoder input.
--max-source-positions 128          # Max length of encoder.
--max-target-positions 1024         # Max length of decoder. If the length of a sample exceeds this limit after up-sampling, the sample will be discarded.
--load-pretrained-model ${data_dir} # Path to a file containing a pre-trained model. It also support a checkpoint file and will automatically convert between lightseq and fairseq architecture.
```

### Decoding Configs


This configs used in the validation. See more configs related to decoding [here](./README.md#decoding).

```bash
--decode-strategy lookahead         # Decoding Strategy. Possible values: greedy, lookahead, beamsearch.
--decode-upsample-scale 8           # Upsampling scale to determine the DAG size during inference.
                                    #   If --upsample-scale used in training is a fixed number, this parameter should be the same value. 
                                    #   If --upsample-scale used in training is a range, this parameter can be the average of the range and optionally tuned after the training."
```

### Criterion Configs

```bash
--criterion nat_dag_loss            # The Criterion for DA-Transformer.
--length-loss-factor  0.1           # Weights on the length prediction loss. Required if --upsample_base "predict" is set.
--max-transition-length 99999       # Specifies the maximum transition distance. A value of -1 indicates no limit, but this cannot be used with CUDA custom operations.
                                    #   To use CUDA operations with no limit, specify a very large number such as 99999.
--glat-p 0.5:0.1@200k               # Set the glancing probability and its annealing schedule. For example, '0.5:0.1@200k' indicates annealing probability from 0.5 to 0.1 in 200k steps.
--glance-strategy number-random     # Set the glancing strategy. Possible values: "number-random" or "None" or "CMLM"
--use-pretrain-loss                 # If true, use the pre-training loss, i.e. the position of segment id will be fixed.
--no-force-emit                     # If true, the position of glanced tokens in the second forward pass will not be fixed.
--torch-dag-loss                    # Use torch native implementation for logsoftmax-gather. It may be slower and consume more GPU memory.
--torch-dag-best-alignment          # Use torch native implementation for dag-best-alignment. It may be slower and consume more GPU memory.
--torch-dag-logsoftmax-gather       # Use torch native implementation for dag-loss. It may be slower and consume more GPU memory.
```

### Optimizer Configs

```bash
--optimizer adam                  # The optimizer. You can use "ls_adam" instead to enable LightSeq's Optimizer
```

### Training Configs

```bash
--max-tokens 4096                 # Specifies the maximum number of tokens (either source or target) allowed in a single batch during training.
                                  #   This number does not include any padding tokens.
--max-tokens-valid 4096           # Specifies the maximum number of tokens (either source or target) allowed in a single batch during validation.
                                  #   This number does not include any padding tokens.
--update-freq 2                   # Specifies the number of steps of gradient accumulation before updating the model.
                                  #     The actual batch size is: GPU number * max_tokens * update_freq.
--max-tokens-after-upsample       # If enabled, the maximum number of tokens (--max-tokens) considered during generation
                                  #   will take into account the upsampling ratio. In other words, the length of the generated sequence will be capped at
                                  #   max(source_length, decoder_length * upsample_scale). Default: False.
--batch-split-by-src 32767        # If this value is greater than 0, it splits a batch into multiple smaller batches.
                                  #   The split is based on the number of source tokens in each batch (considering padding tokens),
                                  #   ensuring that no batch has more source tokens than the specified value.
                                  #   This is different from --update-freq because it works on each GPU separately. It's useful when out-of-memory (OOM) errors occur rarely
                                  #   and you do not want to set a smaller batch size.
--max-encoder-batch-tokens 20000  # Specifies the maximum number of tokens for the encoder input to avoid running out of memory. The default value of None indicates no limit.
--max-decoder-batch-tokens 20000  # Specifies the maximum number of tokens for the decoder input to avoid running out of memory. The default value of None indicates no limit.
```

### Validation Configs

```bash
--eval-bleu                       # Evaluate BLEU scores during validation
--eval-bleu-detok space           # Detokenizer used in BLEU evaluation
--eval-bleu-remove-bpe            # Whether remove bpe in BLEU evaluation
--eval-bleu-print-samples         # Print several samples in BLEU evaluation
--eval-bleu-order 4               # The order of n-gram in BLEU evaluation
```
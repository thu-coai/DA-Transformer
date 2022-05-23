# DA-Transformer

Official Implementation for the ICML2022 paper "**[Directed Acyclic Transformer for Non-Autoregressive Machine Translation](https://arxiv.org/abs/2205.07459)**".

DA-Transformer utilizes a Directed Acyclic Graph (DAG) to capture multiple possible references, where the whole graph can be predicted non-autoregressively. Specifically, our decoder predicts the token probability and transition probability based on the vertex representation, and then follows the transitions to generate a possible translation. When training on the raw data of MT benchmarks, DA-Transformer is the first non-iterative NAT that achieves comparable performance with the vanilla autoregressive Transformer while preserving a 7x~14x latency speedup in inference.

![model](model.png)

## Working in Progress

We are preparing the codes for release, please stay tuned!

## Citing

Please kindly cite us if you find this paper useful.

```
@inproceedings{huang2022DATransformer,
  author = {Fei Huang and Hao Zhou and Yang Liu and Hang Li and Minlie Huang},
  title = {Directed Acyclic Transformer for Non-Autoregressive Machine Translation},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning, {ICML} 2022},
  year = {2022}
}
```
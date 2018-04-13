# suboptimality
Code for paper
*Inference Suboptimality in Variational Autoencoders.*
[[arxiv](https://arxiv.org/abs/1801.03558)]

## Requirements
* `python 3`
* `pytorch`

## Training
`python run.py --train --dataset <dataset> (--lr-schedule --warmup --early-stopping and model args...)`

**model args** are parameters specific to model config, e.g. number of flow steps, wider encoder.

## Evaluation
* IWAE: `python run.py --eval-iwae --dataset <dataset> --eval-path <ckpt path> (model args...)`
* AIS: `python run.py --eval-ais --dataset <dataset> --eval-path <ckpt path> (model args...)`
* Local FFG: `python local_ffg.py --dataset <dataset> --eval-path <ckpt path> (model args...)`
* Local Flow: `python local_flow.py --dataset <dataset> --eval-path <ckpt path> (model args...)`
* BDMC: `python bdmc.py --eval-path <ckpt path> --n-ais-iwae <num samples> --n-ais-dist <num dist> (model args...)`

## Citation
If you use our code, please consider cite the following:
Chris Cremer, Xuechen Li, David Duvenaud
Inference Suboptimality in Variational Autoencoders.
*ICML*, 2018 (to appear).

```
@inproceedings{cremer2018inference,
  author={Cremer, Chris and Li, Xuechen and Duvenaud, David},
  title={Inference Suboptimality in Variational Autoencoders},
  booktitle={ICML},
  year={2018}
}
```

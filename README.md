# suboptimality
Code for paper
*Inference Suboptimality in Variational Autoencoders.*
[[arxiv](https://arxiv.org/abs/1801.03558)]

## Requirements
* `py3`
* `pytorch`

## Training
`python run.py --train --dataset <dataset> (--lr-schedule) (--warmup) (--early-stopping)`

## Evaluation
* IWAE: `python run.py --eval-iwae --dataset <dataset> --eval-path <ckpt path> (model args...)`
* AIS: `python run.py --eval-ais --dataset <dataset> --eval-path <ckpt path> (model args...)`
* Local FFG: `python local_ffg.py --dataset <dataset> --eval-path <ckpt path> (model args...)`
* Local Flow: `python local_flow.py --dataset <dataset> --eval-path <ckpt path> (model args...)`
* BDMC: `python bdmc.py --eval-path <ckpt path> --n-ais-iwae <num samples> --n-ais-dist <num dist>`

## Citation
If you use our code, please consider cite the following:
Chris Cremer, Xuechen Li, David Duvenaud
Inference Suboptimality in Variational Autoencoders.
*ICML*, 2018 (to appear).

```
@inproceedings{cremer2018inference,
  author    = {Aidan N. Gomez and Mengye Ren and Raquel Urtasun and Roger B. Grosse},
  author={Cremer, Chris and Li, Xuechen and Duvenaud, David},
  booktitle={ICML},
  year={2018}
}
```

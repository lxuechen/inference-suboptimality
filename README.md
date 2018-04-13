# inference-suboptimality
Code regarding evaluation for paper
*Inference Suboptimality in Variational Autoencoders.*
[[arxiv](https://arxiv.org/abs/1801.03558)]

## Dependencies
* `python3`
* `pytorch`

## Training
To train on MNIST and Fashion, unzip the compressed files in folder datasets/.
`python run.py --train --dataset <dataset> (--lr-schedule --warmup --early-stopping)`


## Evaluation
* IWAE: `python run.py --eval-iwae --dataset <dataset> --eval-path <ckpt path>`
* AIS: `python run.py --eval-ais --dataset <dataset> --eval-path <ckpt path>`
* Local FFG: `python local_ffg.py --dataset <dataset> --eval-path <ckpt path>`
* Local Flow: `python local_flow.py --dataset <dataset> --eval-path <ckpt path>`
* BDMC: `python bdmc.py --eval-path <ckpt path> --n-ais-iwae <num samples> --n-ais-dist <num dist>`

## Other Experiments
For decoder size, flow affect amortization, test set gap and other experiments, refer to [this](https://github.com/chriscremer/Inference-Suboptimality).

## Citation
If you use our code, please consider cite the following:
Chris Cremer, Xuechen Li, David Duvenaud.
Inference Suboptimality in Variational Autoencoders.

```
@article{cremer2018inference,
  title={Inference Suboptimality in Variational Autoencoders},
  author={Cremer, Chris and Li, Xuechen and Duvenaud, David},
  journal={arXiv preprint arXiv:1801.03558},
  year={2018}
}
```

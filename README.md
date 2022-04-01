# Repository for [Conditional Image Generation by Conditioning Variational Auto-Encoders](https://openreview.net/forum?id=7MV6uLzOChW), to be presented at ICLR 2022

<p float="left">
<img src="ipa-t=0.85.jpg" height="200px"/>
</p>

Based on the code (and pretrained models) released by [Child et al. (2020)](https://arxiv.org/abs/2011.10650) at https://github.com/openai/vdvae/. Please open an issue or email <wsgh@cs.ubc.ca> if you have any questions/requests

## Python requirements
We used Python 3.8.10. See `requirements.txt` for the required packages (which can be installed with pip using `pip install -r requirements.txt`). For training on multiple GPUs, `mpi4py` is required as well.

VQ-VAE and CoModGAN baselines are implemented in Tensorflow. To run any of them you need to install `tensorflow-gpu==1.15.0`.

## Logging to https://wandb.ai/
Our code logs to the Weights & Biases experiment-tracking infrastructure. To use it, you should create an account at https://wandb.ai/ and set the environment variable `WANDB_ENTITY` to your username (e.g. by running `export WANDB_ENTITY=<your username>`). To avoid logging to wandb, e.g. to avoid having to create an account, add the flag `--unobserve` to your training command.

## Data
Download and preprocess each dataset by running the corresponding script, named as `setup_<dataset name>.sh`.

## Pretrained VD-VAEs
Most of our experiments use VD-VAEs trained by [Child et al. (2020)](https://arxiv.org/abs/2011.10650). You can download all of the chekpoints we use by running `bash download-pretrained.sh`. Feel free to comment out the lines downloading non-required checkpoints.

## Example training commands
We prefix all shown commands with `NO_MPI=1`, which sets the environment variable to prevent the script importing `mpi4py` or using distributed training. To train on multiple GPUs, you should install `mpi4py` and execute the command using, e.g., `mpiexec`.

### CIFAR-10
```
NO_MPI=1 python train.py --hps=cifar10 --lr=0.0002 --n_batch=30 --pretrained_load_dir=pretrained/cifar10-1/
```

### ImageNet-64
```
NO_MPI=1 python train.py --hps=imagenet64 --lr=5e-05 --n_batch=4 --pretrained_load_dir=pretrained/imagenet64/
```

### FFHQ-256
```
NO_MPI=1 python train.py --hps=ffhq256 --lr=0.00015 --n_batch=1 --pretrained_load_dir=pretrained/ffhq256/
```

## Pretrained IPA models
We provide a trained IPA model for each dataset [here](https://drive.google.com/drive/folders/1h899kAZbypGyRf1djWAiWytU0YHz7190?usp=sharing). These are trained for the time and number of iterations reported in Table 3 of [the paper](https://openreview.net/forum?id=7MV6uLzOChW).

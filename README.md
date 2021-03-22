# spatially unbiased ddpm
iccv id 1935

## Installation

First please install `tensorfn`

```bash
pip install tensorfn
```

It is simple convenience library for machine learning experiments. Sorry for the inconvenience.

## Training

First prepare lmdb dataset:

```bash
python prepare_data.py --size [SIZES, e.g. 128,256] --out [LMDB NAME] [DATASET PATH]
```

Then run training loop!


```bash
python train.py --conf config/diffusion.conf 
```

Modify path and hyperparameters in diffusion.conf for your own setting.
Set self.pe and self.pe_enc in model.py for positional encoding.

## Generate

For both random generation and reconstruction, run

```bash
python generate.py --conf config/diffusion.conf 
```

Modify **mode** in generate.py: generate or recon
Modify ckpt_diff and refpath for your own setting


## Toy dataset
Spatially biased color-MNIST: https://drive.google.com/drive/folders/15Ll4P0kzLu98ZyJFfFY9vAbSIwoN7GIn?usp=sharing

Code mostly based on https://github.com/rosinality/denoising-diffusion-pytorch


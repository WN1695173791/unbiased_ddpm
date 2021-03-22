import os

import torch
from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision import transforms, utils
from tensorfn import load_arg_config
from tensorfn import distributed as dist
import numpy as np

from model import UNet
from diffusion import GaussianDiffusion, make_beta_schedule
from config import DiffusionConfig

from PIL import Image
from tqdm import tqdm
import random
import numpy as np


def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


def generate(conf, ema, diffusion, device, ckpt_diff):
    mode = 'recon'  # 'generate'

    pbar = range(10)
    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

    resolution = conf.dataset.resolution
    n_sample = conf.evaluate.n_sample

    shape = (n_sample, 3, resolution, resolution)

    refpath = f'reference'
    paths = list(os.listdir(refpath))
    paths = sorted(paths)

    with torch.no_grad():
        ema.eval()

        for i in pbar:
            if mode == 'recon':
                img = np.asarray(Image.open(refpath + '/' + paths[i]))

                if img.shape[2] == 4:
                    img = img[:, :, :3]
                im = img.transpose((2, 0, 1))
                img = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu',
                                   requires_grad=True).cuda() / 127.5 - 1.
                img = F.interpolate(img.unsqueeze(0), size=(resolution, resolution), mode="bicubic", align_corners=True)
                img = torch.roll(img, resolution // 2, 2)
                utils.save_image(
                    img,
                    f'sample/recon_ours/{str(i).zfill(3)}.png',
                    nrow=1,
                    normalize=True,
                    range=(-1, 1)
                )
                img = img.to(device).repeat(n_sample, 1, 1, 1)
                samples = []
                for k in range(9):
                    sample = diffusion.p_recon_loop(ema, shape, device, t=100 * (k + 1), ref=img, roll_pix=resolution // 2)
                    samples.append(sample)
                sample = torch.cat(samples, 0)
            else:
                sample = diffusion.p_sample_loop(ema, shape, device, roll_pix=0)

            for j in range(sample.shape[0]):
                utils.save_image(
                    sample[j].unsqueeze(0),
                    f'sample/recon_ours/{str(i).zfill(3)}_{str(j).zfill(3)}.png',
                    nrow=1,
                    normalize=True,
                    range=(-1, 1)
                )


def main(conf):
    device = "cuda"
    beta_schedule = "linear"
    beta_start = 1e-4
    beta_end = 2e-2
    n_timestep = 1000

    ckpt_diff = f'checkpoint/biased_mnist_pe_enc/diffusion_700000.pt'
    # biased_mnist_pe_enc
    # biased_mnist_baseline
    conf.distributed = False

    ema = conf.model.make()
    ema = ema.to(device)

    print(f'load model from: {ckpt_diff}')
    ckpt = torch.load(ckpt_diff, map_location=lambda storage, loc: storage)
    ema.load_state_dict(ckpt["ema"])

    betas = conf.diffusion.beta_schedule.make()

    # betas: 0.0001 --> 0.02
    diffusion = GaussianDiffusion(betas).to(device)

    generate(conf, ema, diffusion, device, ckpt_diff)


if __name__ == "__main__":
    conf = load_arg_config(DiffusionConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )
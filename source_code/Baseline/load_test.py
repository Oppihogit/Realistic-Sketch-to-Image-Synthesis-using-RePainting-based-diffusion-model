import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import time
import datetime
from models import *
from datasets import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from pix2pixBEGAN import UNet

G_losses = []
D_losses = []


def testing(
    data_dir,
    b_to_a=False,
    init_epoch=100,
    dataset_name="car",
    img_width=64,
    img_height=64,
    samples_per_image=4,
):
    A = "B" if b_to_a else "A"
    B = "A" if b_to_a else "B"
    os.makedirs("test_images_began/%s" % dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    generator = Generator()
    # discriminator = UNet()
    discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    generator.load_state_dict(
        torch.load("saved_models/%s/generator_%d.pth" % (dataset_name, init_epoch))
    )
    discriminator.load_state_dict(
        torch.load("saved_models/%s/discriminator_%d.pth" % (dataset_name, init_epoch))
    )
    # # Apply dropout during inference to increase diversity
    # generator.eval()

    # def apply_dropout(m):
    #     if type(m) == torch.nn.Dropout:
    #         m.train()

    # generator.apply(apply_dropout)

    transforms_ = [
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    val_dataloader = DataLoader(
        ImageDataset(data_dir, transforms_=transforms_, mode="val"),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # def sample_images(batches_done):
    #     imgs = next(iter(val_dataloader))
    #     real_A = Variable(imgs[A].type(Tensor))
    #     real_B = Variable(imgs[B].type(Tensor))
    #     fake_B = generator(real_A)
    #     img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    #     save_image(
    #         img_sample,
    #         "test_images_began/%s/%s.png" % (dataset_name, batches_done),
    #         nrow=5,
    #         normalize=True,
    #     )

    # # print(type(list(path.iterdir())))
    # for i in range(len(list(path.iterdir()))):
    #     sample_images(i)

    def sample_images(batches_done, real_A, real_B):

        # Generate multiple samples for each image
        samples = []
        real_A_stack = [real_A.data] * 4
        for _ in range(samples_per_image):
            noise = torch.randn_like(real_A) * 0.1  # Generate noise with the same shape as real_A
            noisy_real_A = real_A + noise  # Add noise to real_A
            fake_B = generator(noisy_real_A)  # Generate the image with noise
            samples.append(fake_B)
        # Stack generated samples vertically
        real_A_stack = torch.cat(real_A_stack)
        fake_B_stack = torch.cat(samples)
        # Concatenate sketch, generated samples, and real image vertically
        # img_sample = torch.cat((real_A.data, fake_B_stack, real_B.data), 2)
        # img_sample = torch.cat((real_A.data, fake_B_stack), 2)
        img_sample = torch.cat((fake_B_stack, real_A_stack), 2)
        torch.save(img_sample, f"test_images_gan/{dataset_name}/{batches_done}.pt")
        save_image(
            img_sample,
            f"test_images_gan/{dataset_name}/{batches_done}.png",
            normalize=True,
        )

    for i, batch in enumerate(val_dataloader):
        real_A = Variable(batch[A].type(Tensor))
        real_B = Variable(
            batch[B].type(Tensor)
        )  # This is the real image corresponding to the sketch
        sample_images(i, real_A, real_B)


if __name__ == "__main__":
    path = Path("test_selected/combined/car")
    testing(data_dir=path)

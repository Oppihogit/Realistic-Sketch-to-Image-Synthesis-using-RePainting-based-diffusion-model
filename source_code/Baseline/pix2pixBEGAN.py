import os
import numpy as np
import math
import itertools
import sys
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

G_losses = []
D_losses = []


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class UNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3

        # input: 64x64xin_channels
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )  # pool to 32x32x64

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )  # pool to 16x16x128

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )  # pool to 8x8x256

        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )  # pool to 4x4x512

        self.bottom = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )  # out: 8x8x256

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )  # out: 16x16x128

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )  # out: 32x32x64

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )  # out: 64x64x32

        self.out = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)  # 64x64x64
        xp1 = F.max_pool2d(x1, 2)

        x2 = self.down2(xp1)  # 128x32x32
        xp2 = F.max_pool2d(x2, 2)

        x3 = self.down3(xp2)  # 256x16x16
        xp3 = F.max_pool2d(x3, 2)

        x4 = self.down4(xp3)  # 512x8x8
        xp4 = F.max_pool2d(x4, 2)

        x5 = self.bottom(xp4)  # 512x4x4

        # Decoder
        yu4 = self.upconv4(x5)
        y4 = self.up4(torch.cat([yu4, x4], dim=1))

        yu3 = self.upconv3(y4)
        y3 = self.up3(torch.cat([yu3, x3], dim=1))

        yu2 = self.upconv2(y3)
        y2 = self.up2(torch.cat([yu2, x2], dim=1))

        yu1 = self.upconv1(y2)
        y1 = self.up1(torch.cat([yu1, x1], dim=1))

        return self.out(y1)


def training(
    data_dir,
    b_to_a=False,
    init_epoch=0,
    n_epochs=101,
    dataset_name="dog",
    batch_size=1,
    img_width=64,
    img_height=64,
    lr=0.0002,
    b1=0.5,
    b2=0.999,
    num_workers=8,
    lambda_pixel=10,
    sample_interval=3000,
    checkpoint_interval=20,
):
    A = "B" if b_to_a else "A"
    B = "A" if b_to_a else "B"
    os.makedirs("images_began/%s" % dataset_name, exist_ok=True)
    os.makedirs("saved_models_began/%s" % dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    patch = (1, img_height // 2**4, img_width // 2**4)

    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    generator = Generator()
    discriminator = UNet()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()

    if init_epoch != 0:
        generator.load_state_dict(
            torch.load("saved_models_began/%s/generator_%d.pth" % (dataset_name, init_epoch))
        )
        discriminator.load_state_dict(
            torch.load("saved_models_began/%s/discriminator_%d.pth" % (dataset_name, init_epoch))
        )
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    transforms_ = [
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ImageDataset(data_dir, transforms_=transforms_),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_dataloader = DataLoader(
        ImageDataset(data_dir, transforms_=transforms_, mode="val"),
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def sample_images(batches_done):
        imgs = next(iter(val_dataloader))
        real_A = Variable(imgs[A].type(Tensor))
        real_B = Variable(imgs[B].type(Tensor))
        fake_B = generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(
            img_sample,
            "images_began/%s/%s.png" % (dataset_name, batches_done),
            nrow=5,
            normalize=True,
        )

    # def plot_metrics(G_losses, D_losses):
    #     plt.figure(figsize=(8, 5))
    #     plt.title("Generator and Discriminator Loss During Training")
    #     plt.plot(G_losses, label="G")
    #     plt.plot(D_losses, label="D")
    #     plt.xlabel("iterations")
    #     plt.ylabel("Loss")
    #     plt.legend()
    #     plt.show()

    #  Training
    # Initialize k to control the balance between generator and discriminator

    k = 0.0
    lambda_k = 0.001
    gamma = 0.5
    M_global = AverageMeter()
    prev_time = time.time()

    for epoch in range(init_epoch, n_epochs):
        for i, batch in enumerate(dataloader):

            # Load real images
            real_A = Variable(batch[A].type(Tensor))
            real_B = Variable(batch[B].type(Tensor))

            # ===== Train Discriminator =====
            optimizer_D.zero_grad()

            # Generate fake images
            fake_B = generator(real_A).detach()  # Detach to avoid training G on these gradients

            # Reconstruct real and fake images
            recon_real = discriminator(real_B)
            recon_fake = discriminator(fake_B)

            # Discriminator loss
            loss_real = torch.mean(torch.abs(recon_real - real_B))
            loss_fake = torch.mean(torch.abs(recon_fake - fake_B))
            loss_D = loss_real - k * loss_fake

            # Update discriminator
            loss_D.backward()
            optimizer_D.step()

            # ===== Train Generator =====
            optimizer_G.zero_grad()

            # Generate fake images for generator update
            fake_B = generator(real_A)

            # Reconstruct generated images
            recon_fake = discriminator(fake_B)

            # Generator loss (invert errD_fake for generator update)
            loss_GAN = torch.mean(torch.abs(recon_fake - fake_B))
            loss_pixel = F.l1_loss(fake_B, real_B)
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            # Update generator
            loss_G.backward()
            optimizer_G.step()

            # ===== Update k to balance training =====
            balance = (gamma * loss_real - loss_G).item()
            k = min(max(k + lambda_k * balance, 0), 1)  # Ensure k is within [0, 1]

            # Record the progress
            M_global.update(loss_real.item() + abs(balance))

            # Optional: code to log the losses, save models, and sample images periodically

            # Note: This is a simplified version and assumes all variables and methods are properly defined.

            batches_done = epoch * len(dataloader) + i
            batches_left = n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f], k: %f, ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    # loss_pixel.item(),
                    k,
                    # loss_GAN.item(),
                    time_left,
                )
            )
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
            if batches_done % sample_interval == 0:
                sample_images(batches_done)
                # plot_metrics(G_losses, D_losses)

        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            torch.save(
                generator.state_dict(),
                "saved_models_began/%s/generator_%d.pth" % (dataset_name, epoch),
            )
            torch.save(
                discriminator.state_dict(),
                "saved_models_began/%s/discriminator_%d.pth" % (dataset_name, epoch),
            )


if __name__ == "__main__":
    path = Path("datasets/all_data_train/combined/dog")
    training(data_dir=path)

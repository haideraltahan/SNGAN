import os
import numpy as np
import fire

from torch.nn.utils import spectral_norm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm

import torch.nn as nn
import torch


class Linear(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Linear, self).__init__()
        self.layer = nn.Linear(in_feat, out_feat)
        nn.init.orthogonal_(self.layer.weight)
        self.layer = spectral_norm(self.layer)

    def forward(self, x):
        return self.layer(x)


class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.img_shape = img_shape
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class Main(object):
    def __init__(self, channels=1, img_size=28, data_folder='data', samples_folder='images', batch_size=128,
                 latent_dim=100, n_cpu=12):
        super(Main, self).__init__()
        img_shape = (channels, img_size, img_size)
        self.latent_dim = latent_dim
        # Initialize generator and discriminator
        self.generator = Generator(img_shape, latent_dim)
        self.discriminator = Discriminator(img_shape)

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

        # Configure data loader
        os.makedirs(samples_folder, exist_ok=True)
        os.makedirs(f"./{data_folder}/mnist", exist_ok=True)
        self.dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                f"./{data_folder}/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_cpu
        )

    def train(self, n_critic=5, lr=0.01, betas=(0.5, 0.9), n_epochs=200, sample_interval=500):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # Logger
        loss_log = tqdm(total=0, position=0, bar_format='{desc}')

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        batches_done = 0
        for epoch in range(n_epochs):
            for i, (imgs, _) in enumerate(self.dataloader):

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                fake_imgs = self.generator(z)

                # Real images
                real_validity = nn.ReLU()(1.0 - self.discriminator(real_imgs)).mean()
                # Fake images
                fake_validity = nn.ReLU()(1.0 + self.discriminator(fake_imgs)).mean()

                # Adversarial loss - Hinge Loss
                d_loss = real_validity + fake_validity

                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_imgs = self.generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    g_loss = -self.discriminator(fake_imgs).mean()

                    g_loss.backward()
                    optimizer_G.step()

                    loss_log.set_description_str(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
                            epoch, n_epochs, i, len(self.dataloader), d_loss.item(), g_loss.item())
                    )

                    if batches_done % sample_interval == 0:
                        save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                    batches_done += n_critic

if __name__ == '__main__':
    fire.Fire(Main)

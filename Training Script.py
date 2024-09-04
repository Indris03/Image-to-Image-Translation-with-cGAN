# train.py

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models.generator import GeneratorUNet
from models.discriminator import Discriminator
import itertools
from torchvision.utils import save_image

# Hyperparameters
batch_size = 1
lr = 0.0002
n_epochs = 200
decay_epoch = 100
img_height = 256
img_width = 256
channels = 3

# Paths
dataset_path = 'datasets/facades/train'
output_path = 'output/'
os.makedirs(output_path, exist_ok=True)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*channels, [0.5]*channels)
])

# Dataset and DataLoader
dataset = ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = GeneratorUNet().cuda()
discriminator = Discriminator().cuda()

# Loss functions
criterion_GAN = nn.MSELoss()
criterion_L1 = nn.L1Loss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
for epoch in range(n_epochs):
    for i, (input, target) in enumerate(dataloader):
        input = input.cuda()
        target = target.cuda()
        
        # Adversarial ground truths
        valid = torch.ones((input.size(0), 1, 30, 30)).cuda()
        fake = torch.zeros((input.size(0), 1, 30, 30)).cuda()
        
        # ------------------
        #  Train Generator
        # ------------------
        optimizer_G.zero_grad()
        
        # Generate a batch of images
        gen_output = generator(input)
        
        # Loss measures generator's ability to fool the discriminator
        pred_fake = discriminator(gen_output, input)
        loss_GAN = criterion_GAN(pred_fake, valid)
        
        # L1 loss
        loss_L1 = criterion_L1(gen_output, target) * 100
        
        # Total loss
        loss_G = loss_GAN + loss_L1
        loss_G.backward()
        optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Real loss
        pred_real = discriminator(target, input)
        loss_real = criterion_GAN(pred_real, valid)
        
        # Fake loss
        pred_fake = discriminator(gen_output.detach(), input)
        loss_fake = criterion_GAN(pred_fake, fake)
        
        # Total loss
        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()
        
        # Print progress
        print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
              f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")
        
        # Save sample images
        if i % 100 == 0:
            save_image(gen_output.data, os.path.join(output_path, f"gen_{epoch}_{i}.png"), normalize=True)
            save_image(target.data, os.path.join(output_path, f"real_{epoch}_{i}.png"), normalize=True)

    # Save models checkpoints
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), f"models/generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"models/discriminator_epoch_{epoch}.pth")

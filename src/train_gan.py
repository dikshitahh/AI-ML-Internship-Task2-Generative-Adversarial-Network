#import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

#Device Configuration
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Hyperparameter
batch_size = 128
learning_rate = 0.0002
latent_dim = 100
epochs = 50

#load Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
dataset = datasets.MNIST(
    root = "./data",
    train = True,
    transform = transform,
    download = True
)
loader = DataLoader(dataset, batch_size = batch_size,
                    shuffle = True)

#Visualize Dataset
import matplotlib.pyplot as plt
images, labels = next(iter(loader))
plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.axis("off")
plt.show()

#Create Model
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

#Loss Function
criterion = nn.BCELoss()

#Optimizer
optimizer_G = optim.Adam(
    generator.parameters(),
    lr = learning_rate,
    betas = (0.5, 0.999)
)
optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr = learning_rate,
    betas = (0.5, 0.999)
)

#Creating output Folder
os.makedirs("output/generated_samples", exist_ok = True)

#Training Loop
for epoch in range(epochs):

    for i, (real_images, _) in enumerate(loader):

        # Move images to device
        real_images = real_images.to(device)
        batch = real_images.size(0)

        # Creating labels
        real_labels = torch.ones(batch, 1).to(device)
        fake_labels = torch.zeros(batch, 1).to(device)

        # Train Discriminator
        outputs = discriminator(real_images)
        loss_real = criterion(outputs, real_labels)
        noise = torch.randn(batch, latent_dim).to(device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images.detach())
        loss_fake = criterion(outputs, fake_labels)
        d_loss = loss_real + loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        noise = torch.randn(batch, latent_dim).to(device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    # Print every epoch
    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"D Loss: {d_loss.item():.4f} "
        f"G Loss: {g_loss.item():.4f}"
    )

    # Save image every 5 epochs
    if (epoch+1) % 5 == 0:
        save_image(
            fake_images[:25],
            f"output/generated_samples/epoch_{epoch+1}.png",
            nrow=5,
            normalize=True
        )

#Saving the model
  torch.save(generator.state_dict(), "generator.pth")
  torch.save(discriminator.state_dict(), "discriminator.pth")

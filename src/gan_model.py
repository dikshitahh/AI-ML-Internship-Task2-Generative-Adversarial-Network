# Build Generator
class Generator(nn.Module):
  def __init__(self, latent_dim):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(True),
        nn.Linear(256,512),
        nn.ReLU(True),
        nn.Linear(512, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 784),
        nn.Tanh()
    )
  def forward(self, x):
        img = self.model(x)
        img = img.view(x.size(0), 1, 28, 28)
        return img

  #Build Discriminator
  class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.model(img)

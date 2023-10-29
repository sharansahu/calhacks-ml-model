import logging
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from dpwgan.utils import create_mnist_gan, estimate_epochs_from_sample

# Parameters for GAN
noise_dim = 100
image_channels = 1  # Grayscale images
image_size = 28     # Original MNIST image size is 28x28

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalizing the images
])
mnist_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transform, download=True)
large_dataset = Subset(mnist_dataset, range(12000))
dataloader = DataLoader(large_dataset, batch_size=64, shuffle=True)

# Create the GAN (adjust architecture and noise function as needed)
gan = create_mnist_gan(noise_dim)

estimated_epochs = estimate_epochs_from_sample(dataloader)

data_tensor = torch.cat([images for images, _ in dataloader], dim=0)
logging.basicConfig(level=logging.INFO)
gan.train(data=data_tensor,
          epochs=5,
          n_critics=5,
          batch_size=64,
          learning_rate=1e-4,
          weight_clip=0.01,
          sigma=1)  # sigma can be set for differential privacy, if desired

# Generate synthetic images
num_synthetic_images = 5
synthetic_images = gan.generate(num_synthetic_images)
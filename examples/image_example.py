from torchvision import transforms
import logging
import torch
from torch.utils.data import DataLoader
import os
from dpwgan.datasets import ImageDataset_ResNet

from dpwgan.utils import build_gan_for_image_size_pretrained


# Set your dataset directory
dataset_directory = "../CT_Medical_Image_Dataset/tiff_images"

# Create the dataset
dataset = ImageDataset_ResNet(
    directory=dataset_directory,
    grayscale=True,  # Assuming medical images are grayscale
    resize_and_pad_to_power_of_two=True  # or resize_to_224=True for ResNet compatibility
)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Set the image size (e.g., 224 for ResNet compatibility or the power of two size)
transformed_image_size = dataset.get_transformed_image_size()

# Build the GAN
gan = build_gan_for_image_size_pretrained(
    image_size=transformed_image_size,
    noise_dim=100,
    image_channels=1,  # Set to 1 for grayscale images
    use_pretrained=True
)

data_tensor = torch.cat([images for images in data_loader], dim=0)

logging.basicConfig(level=logging.INFO)
gan.train(
    data_tensor,
    epochs=10,
    n_critics=5,
    batch_size=64,
    learning_rate=1e-4,
    sigma=0.1,  # Adjust for differential privacy needs
    weight_clip=0.01
)

# Generate synthetic images
num_synthetic_images = 5
synthetic_images = gan.generate(num_synthetic_images)
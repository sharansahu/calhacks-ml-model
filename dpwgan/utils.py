import pandas as pd
import torch
import numpy as np
import math
import torchvision

from dpwgan import DPWGAN, MultiCategoryGumbelSoftmax



def create_categorical_gan(noise_dim, hidden_dim, output_dims):
    generator = torch.nn.Sequential(
        torch.nn.Linear(noise_dim, hidden_dim),
        torch.nn.ReLU(),
        MultiCategoryGumbelSoftmax(hidden_dim, output_dims)
    )

    discriminator = torch.nn.Sequential(
        torch.nn.Linear(sum(output_dims), hidden_dim),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(hidden_dim, 1)
    )

    def noise_function(n):
        return torch.randn(n, noise_dim)

    gan = DPWGAN(
        generator=generator,
        discriminator=discriminator,
        noise_function=noise_function
    )

    return gan

def calculate_layer_dimensions(image_size):
    """
    Calculates the number of layers and initial feature sizes for the generator and discriminator
    based on the input image size.
    """
    layers = max(int(math.log2(image_size)) - 2, 1)  # Fewer layers for MNIST
    gen_initial_feature_size = 64 * (2 ** (layers - 1))  # We start big and get smaller for the generator
    disc_initial_feature_size = 64  # Start small and get bigger for discriminator

    return layers, gen_initial_feature_size, disc_initial_feature_size

def create_mnist_gan(noise_dim=100):
    # Generator for MNIST
    generator = torch.nn.Sequential(
        torch.nn.Linear(noise_dim, 128 * 7 * 7),
        torch.nn.ReLU(True),
        torch.nn.Unflatten(1, (128, 7, 7)),
        torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(True),
        torch.nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),  # Output channel = 1 for grayscale
        torch.nn.Tanh()  # Output: 1 x 28 x 28
    )

    # Discriminator for MNIST
    discriminator = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, 4, 2, 1, bias=False),  # Input channel = 1 for grayscale
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False),
        torch.nn.Flatten(),
        torch.nn.Linear(128 * 7 * 7, 1),
        torch.nn.Sigmoid()
    )

    def noise_function(n):
        return torch.randn(n, noise_dim)

    gan = DPWGAN(generator, discriminator, noise_function)
    return gan

def build_gan_for_image_size(image_size, noise_dim=100, image_channels=3):
    layers = 512
    gen_layers = [
        torch.nn.Linear(noise_dim, layers * 4 * 4),
        torch.nn.ReLU(True),
        torch.nn.Unflatten(1, (layers, 4, 4))
    ]

    current_size = 4
    while current_size < image_size:
        gen_layers.extend([
            torch.nn.ConvTranspose2d(layers, layers // 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(layers // 2),
            torch.nn.ReLU(True)
        ])
        layers //= 2
        current_size *= 2

    gen_layers.extend([
        torch.nn.ConvTranspose2d(layers, image_channels, 4, 2, 1, bias=False),
        torch.nn.Tanh()
    ])

    generator = torch.nn.Sequential(*gen_layers)

    # Discriminator
    layers = image_channels
    disc_layers = []
    current_size = image_size
    while current_size > 4:
        disc_layers.extend([
            torch.nn.Conv2d(layers, layers * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(layers * 2),
            torch.nn.LeakyReLU(0.2, inplace=True)
        ])
        layers *= 2
        current_size //= 2

    disc_layers.extend([
        torch.nn.Flatten(),
        torch.nn.Linear(layers // 2 * 4 * 4, 1),
        torch.nn.Sigmoid()
    ])

    discriminator = torch.nn.Sequential(*disc_layers)

    def noise_function(n):
        return torch.randn(n, noise_dim)

    gan = DPWGAN(
        generator=generator,
        discriminator=discriminator,
        noise_function=noise_function
    )

    return gan

def build_gan_for_image_size_pretrained(image_size, noise_dim=100, image_channels=3, use_pretrained=False):
    # Generator setup
    layers = 512
    gen_layers = [
        torch.nn.Linear(noise_dim, layers * 4 * 4),
        torch.nn.ReLU(True),
        torch.nn.Unflatten(1, (layers, 4, 4))
    ]

    current_size = 4
    while current_size < image_size:
        gen_layers.extend([
            torch.nn.ConvTranspose2d(layers, layers // 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(layers // 2),
            torch.nn.ReLU(True)
        ])
        layers //= 2
        current_size *= 2

    gen_layers.extend([
        torch.nn.ConvTranspose2d(layers, image_channels, 4, 2, 1, bias=False),
        torch.nn.Tanh()
    ])

    generator = torch.nn.Sequential(*gen_layers)

    # Discriminator setup using a ResNet-like architecture
    discriminator = torchvision.models.resnet18(pretrained=False)

    # Modify the first convolutional layer to accept grayscale images
    if image_channels == 1:
        original_conv1 = discriminator.conv1
        discriminator.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            discriminator.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

    discriminator.fc = torch.nn.Linear(discriminator.fc.in_features, 1)

    if use_pretrained:
        # Load pre-trained weights
        pretrained_model = torchvision.models.resnet18(pretrained=True)
        # Transfer weights excluding the first conv and final fully connected layer
        model_dict = discriminator.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict and "fc" not in k and "conv1" not in k}
        model_dict.update(pretrained_dict)
        discriminator.load_state_dict(model_dict, strict=False)

    def noise_function(n):
        return torch.randn(n, noise_dim)

    # Creating the GAN object
    gan = DPWGAN(
        generator=generator,
        discriminator=discriminator,
        noise_function=noise_function
    )

    return gan


def estimate_epochs_from_sample(dataloader, base_epochs=20):
    """
    Estimates the number of training epochs based on a sample image from the dataloader.

    Parameters:
    - dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    - base_epochs (int): Base number of epochs for simple images.

    Returns:
    - int: An estimated number of epochs for training.
    """
    sample_image, _ = next(iter(dataloader))

    # Analyze the image complexity
    std_dev = torch.std(sample_image).item()
    complexity_index = np.clip(std_dev, 0.5, 1.0)  # Adjust these limits based on experimentation

    # Adjust epochs based on complexity
    adjusted_epochs = int(base_epochs * complexity_index * 2)  # Multiply by 2 for a range of [base_epochs, 2 * base_epochs]

    return adjusted_epochs

def percentage_crosstab(variable_one, variable_two):
    return 100*pd.crosstab(variable_one, variable_two).apply(
        lambda r: r/r.sum(), axis=1
    )

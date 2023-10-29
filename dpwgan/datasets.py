import collections

import numpy as np
import pandas as pd
import torch
import os
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from torch.utils.data import Dataset
import math
import cv2

def to_onehot(data, codes):
    indices = [np.where(codes == val)[0][0] for val in data]
    indices = torch.LongTensor(list([val] for val in indices))
    onehot = torch.FloatTensor(indices.size(0), len(codes)).zero_()
    onehot.scatter_(1, indices, 1)
    return onehot


def from_onehot(data, codes):
    return codes[[np.where(data[i] == 1)[0][0] for i in range(len(data))]]

class ImageDataset(Dataset):
    """
    A PyTorch Dataset for loading and transforming images from a specified directory,
    with the option to resize and pad them to the nearest power of two dimensions
    while maintaining the aspect ratio. Includes error handling for corrupt or non-image files.

    Parameters
    ----------
    directory : str
        The path to the directory containing the image files.
    transform : torchvision.transforms.Compose, optional
        A torchvision transforms composition to apply to each image. If not provided,
        a default transformation (resizing and normalizing pixel values) is applied.
    grayscale : bool, optional
        If True, images will be converted to grayscale. Otherwise, images are assumed to be in RGB.
        Default is False.
    resize_and_pad_to_power_of_two : bool, optional
        If True, images will be resized and padded to the nearest power of two dimensions
        while maintaining aspect ratio. Default is False.

    Attributes
    ----------
    directory : str
        Directory containing image files.
    image_files : list
        List of filenames in the directory that could be successfully loaded as images.
    grayscale : bool
        Flag indicating whether images are converted to grayscale.
    resize_and_pad_to_power_of_two : bool
        Flag indicating whether images are resized and padded to power of two dimensions.
    transform : torchvision.transforms.Compose
        Transformations to be applied to each image.
    """

    def __init__(self, directory, transform=None, grayscale=False, resize_and_pad_to_power_of_two=False):
        self.directory = directory
        self.grayscale = grayscale
        self.resize_and_pad_to_power_of_two = resize_and_pad_to_power_of_two

        # Filter out non-image files and handle corrupt images
        self.image_files = []
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            try:
                with Image.open(file_path) as img:
                    self.image_files.append(file_name)
            except (UnidentifiedImageError, FileNotFoundError, IOError):
                print(f"Warning: Skipping file {file_path} due to loading error.")

        if not transform:
            # Apply default transformations (normalizing to [-1, 1] range)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) if self.grayscale else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def nearest_power_of_two(self, n):
        power = round(math.log(n, 2))
        return 2 ** power

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.directory, self.image_files[idx])
        try:
            with Image.open(image_path) as img:
                if self.grayscale:
                    img = img.convert('L')
                
                if self.resize_and_pad_to_power_of_two:
                    original_width, original_height = img.size
                    new_width = self.nearest_power_of_two(original_width)
                    new_height = self.nearest_power_of_two(original_height)

                    # Resize image to the longer side's nearest power of two
                    longer_side = max(new_width, new_height)
                    img = img.resize((longer_side, longer_side), Image.ANTIALIAS)

                    # Padding to maintain aspect ratio
                    delta_width = longer_side - original_width
                    delta_height = longer_side - original_height
                    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
                    img = ImageOps.expand(img, padding)

                return self.transform(img)
        except Exception as e:
            print(f"Error processing file {image_path}: {e}")
            return None

class ImageDataset_ResNet(Dataset):
    """
    A PyTorch Dataset for loading and transforming images from a specified directory.
    This class allows for optional resizing to a fixed size (e.g., 224x224 for compatibility with ResNet models)
    or resizing and padding to the nearest power of two dimensions while maintaining the aspect ratio.
    It includes error handling for corrupt or non-image files.

    Parameters
    ----------
    directory : str
        The path to the directory containing the image files.
    transform : torchvision.transforms.Compose, optional
        A torchvision transforms composition to apply to each image. If not provided,
        a default transformation (resizing and normalizing pixel values) is applied.
    grayscale : bool, optional
        If True, images will be converted to grayscale. Otherwise, images are assumed to be in RGB.
        Default is False.
    resize_and_pad_to_power_of_two : bool, optional
        If True, images will be resized and padded to the nearest power of two dimensions
        while maintaining aspect ratio. Default is False.
    resize_to_224 : bool, optional
        If True, images will be resized to 224x224 pixels. This is useful for models
        pre-trained on ImageNet. Default is False.

    Attributes
    ----------
    directory : str
        Directory containing image files.
    image_files : list
        List of filenames in the directory that could be successfully loaded as images.
    grayscale : bool
        Flag indicating whether images are converted to grayscale.
    resize_and_pad_to_power_of_two : bool
        Flag indicating whether images are resized and padded to power of two dimensions.
    resize_to_224 : bool
        Flag indicating whether images are resized to 224x224 pixels.
    transform : torchvision.transforms.Compose
        Transformations to be applied to each image.
    """

    def __init__(self, directory, transform=None, grayscale=False, resize_and_pad_to_power_of_two=False, resize_to_224=False):
        self.directory = directory
        self.grayscale = grayscale
        self.resize_and_pad_to_power_of_two = resize_and_pad_to_power_of_two
        self.resize_to_224 = resize_to_224

        # Filter out non-image files and handle corrupt images
        self.image_files = []
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            try:
                # Try loading with OpenCV
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.image_files.append(file_name)
                else:
                    raise IOError(f"Unable to load image: {file_path}")
            except Exception as e:
                print(f"Warning: Skipping file {file_path} due to loading error. Error: {e}")

        # Apply transformations
        if not transform:
            if self.resize_to_224:
                # Apply resizing to 224x224 and standard ImageNet normalization
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Apply default transformations (normalizing to [-1, 1] range)
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) if self.grayscale else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    def nearest_power_of_two(self, n):
        power = round(math.log(n, 2))
        return 2 ** power

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.directory, self.image_files[idx])
        try:
            # Load image with OpenCV
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            # Check if color space conversion is needed
            if self.grayscale:
                # If the image is not already grayscale
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) < 3:
                # If a color image is expected but a grayscale image is found
                raise ValueError("Expected a color image but found grayscale.")

            # Convert OpenCV image to PIL image for further processing
            img = Image.fromarray(img)

            if self.resize_to_224:
                img = img.resize((224, 224), Image.ANTIALIAS)
            elif self.resize_and_pad_to_power_of_two:
                original_width, original_height = img.size
                new_width = self.nearest_power_of_two(original_width)
                new_height = self.nearest_power_of_two(original_height)

                longer_side = max(new_width, new_height)
                img = img.resize((longer_side, longer_side), Image.ANTIALIAS)

                delta_width = longer_side - original_width
                delta_height = longer_side - original_height
                padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
                img = ImageOps.expand(img, padding)

            return self.transform(img)
        except Exception as e:
            print(f"Error processing file {image_path}: {e}")
            return None

    def get_transformed_image_size(self):
        # If the images are resized to 224x224
        if self.resize_to_224:
            return 224
        
        # If the images are resized and padded to the nearest power of two
        if self.resize_and_pad_to_power_of_two:
            # We assume all images are of the same dimensions, so we take the dimensions of the first image
            first_image_path = os.path.join(self.directory, self.image_files[0])
            img = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
            if self.grayscale and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Calculate nearest power of two for both dimensions
            width_power_of_two = self.nearest_power_of_two(img.shape[1])
            height_power_of_two = self.nearest_power_of_two(img.shape[0])
            # Return the larger of the two dimensions
            return max(width_power_of_two, height_power_of_two)

        # Default return if no resizing is applied
        return None


class CategoricalDataset(object):
    """Class to convert between pandas DataFrame with categorical variables
    and a torch Tensor with onehot encodings of each variable

    Parameters
    ----------
    data : pandas.DataFrame
    """
    def __init__(self, data):
        self.data = data
        self.codes = collections.OrderedDict(
            (var, np.unique(data[var])) for var in data
        )
        self.dimensions = [len(code) for code in self.codes.values()]

    def to_onehot_flat(self):
        """Returns a torch Tensor with onehot encodings of each variable
        in the original data set

        Returns
        -------
        torch.Tensor
        """
        return torch.cat([to_onehot(self.data[var], code)
                          for var, code
                          in self.codes.items()], 1)

    def from_onehot_flat(self, data):
        """Converts from a torch Tensor with onehot encodings of each variable
        to a pandas DataFrame with categories

        Parameters
        ----------
        data : torch.Tensor

        Returns
        -------
        pandas.DataFrame
        """
        categorical_data = pd.DataFrame()
        index = 0
        for var, code in self.codes.items():
            var_data = data[:, index:(index+len(code))]
            categorical_data[var] = from_onehot(var_data, code)
            index += len(code)
        return categorical_data

# calhacks-ml-model

# DPWGAN
Code for a differentially private Wasserstein GAN implemented in [PyTorch](https://pytorch.org/)

## Installation

This package requires Python >= 3.5

## Usage

Setting up, training, and generating from a DPWGAN:

```
gan = DPWGAN(generator, discriminator, noise_function)
gan.train(data)
synthetic_data = gan.generate(100)
```

`generator` and `discriminator` should be `torch.nn` modules, and
`noise_function` should generate random data as input to the `generator`.
As a simple example:

```
# simple generator module
generator = torch.nn.Sequential(
    torch.nn.Linear(noise_dim, hidden_dim),
    torch.nn.ReLU(),
    MultiCategoryGumbelSoftmax(hidden_dim, output_dims)
)

# simple discriminator module
discriminator = torch.nn.Sequential(
    torch.nn.Linear(sum(output_dims), hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, 1)
)

# simple noise function (input to generator module)
def noise_function(n):
    return torch.randn(n, noise_dim)
```

The [`examples`](dpwgan/examples) folder has four scripts to demonstrate setting
up, training, and generating data with a DPWGAN with categorical data sets.

[`simple_example.py`](dpwgan/examples/simple_example.py) shows how to create
a generator, discriminator, and noise function, and applies the DPWGAN
to a toy categorical data set.

[`image_example.py`](dpwgan/examples/simple_example.py) applies a DPWGAN to
CT scan data. Download the data set at (https://www.kaggle.com/datasets/kmader/siim-medical-images/data).

[`mnist_example.py`](dpwgan/examples/mnist_example.py) applies a DPWGAN to
MNIST data.

If you want to print the training losses,
set the logging level to `logging.INFO`:

```
import logging
logging.basicConfig(level=logging.INFO)
```

## Model

This model is largely an implementation of the [Differentially Private Generative Adversarial Network model](https://arxiv.org/abs/1802.06739)
from Xie, Lin, Wang, Wang, and Zhou (2018).

## Calculating epsilon

You can use the [compute_dp_sgd_privacy](https://github.com/tensorflow/privacy/blob/979748e09c416ea2d4f85e09b033aa9aa097ead2/tensorflow_privacy/privacy/analysis/compute_dp_sgd_privacy.py)
script in the [TensorFlow Privacy](https://github.com/tensorflow/privacy)
library to calculate the privacy loss epsilon for a given data set and training regime.
`N` corresponds to the number of data points, and `noise_multiplier` corresponds to `sigma`.
`batch_size` and `epochs` are the same as in training.
`delta` (the tolerance probability for privacy loss beyond `epsilon`)
can be set according to your needs or optimized (the library currently uses `1e-6` as a reasonable default).
The method also gives an estimate of α for (α, ε)-Rényi differential privacy.

Note that calculating the appropriate value for the `weight_clip` is non-trivial,
and depends on the architecture of the discriminator. See section 3.4 of
[Xie, Lin, Wang, Wang, and Zhou (2018)](https://arxiv.org/abs/1802.06739)
for details, where `c_p` corresponds to `weight_clip`.
`sigma` should be set to the product of `sigma_n` and `c_g`.

## Acknowledgements

Starter code is largely based off and modified using https://github.com/civisanalytics/dpwgan

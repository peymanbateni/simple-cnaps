from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import PIL
from PIL import ImageOps

import torchvision.transforms as transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
omniglot_normalize = transforms.Normalize(mean=[0.078705, 0.078705, 0.078705], std=[0.26927784, 0.26927784, 0.26927784])
mnist_normalize = transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
cifar_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


omniglot_transforms = transforms.Compose([
    transforms.Lambda(lambda x: ImageOps.invert(x)),
    transforms.Resize(84, interpolation=PIL.Image.LANCZOS),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    omniglot_normalize
])

quickdraw_transforms = transforms.Compose([
    transforms.Resize(84, interpolation=PIL.Image.LANCZOS),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

imagenet_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize
    ])

mnist_transforms = transforms.Compose([
    transforms.Resize((84, 84), interpolation=PIL.Image.LANCZOS),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    mnist_normalize
])

cifar_transforms = transforms.Compose([
    transforms.Resize((84, 84), interpolation=PIL.Image.LANCZOS),
    transforms.ToTensor(),
    cifar_normalize
])

standard_transforms = transforms.Compose([
    transforms.Resize((84, 84), interpolation=PIL.Image.LANCZOS),
    transforms.ToTensor(),
    normalize
])

cifar_transforms = transforms.Compose([
    transforms.Resize((84, 84), interpolation=PIL.Image.LANCZOS),
    transforms.ToTensor(),
    cifar_normalize
])

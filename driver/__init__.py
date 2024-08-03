import torch
import torchvision.transforms as tvt
# import albumentations as A
from stone_data.auto_augment import AutoAugment
OPTIM = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# train_transforms = tvt.Compose([
#     # transforms.Resize((224, 224)),
#     tvt.ToPILImage(),
#     tvt.RandomAffine([-90, 90], translate=[0.01, 0.1],
#                             scale=[0.9, 1.1]),
#     tvt.RandomRotation((-10, 10)),
#     tvt.RandomHorizontalFlip(),
#     tvt.RandomVerticalFlip(),
#     tvt.ToTensor(),
#     tvt.Normalize(mean=mean,
#                   std=std)
# ])

import random
from PIL import ImageFilter
class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# transform_local = tvt.Compose([
#     tvt.ToPILImage(),
#     tvt.RandomApply(
#         [tvt.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
#     ),
#     tvt.RandomGrayscale(p=0.2),
#     tvt.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
#     tvt.RandomHorizontalFlip(),
#     tvt.ToTensor(),
#     tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
transform_local = tvt.Compose([
    tvt.ToPILImage(),
    AutoAugment(),
    tvt.ToTensor(),
    # tvt.Normalize(mean=mean, std=std)
])

transform_test = tvt.Compose([
    tvt.ToPILImage(),
    # tvt.Resize((224, 224)),
    tvt.ToTensor(),
    # tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

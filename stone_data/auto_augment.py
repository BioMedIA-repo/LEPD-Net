# -*- coding: utf-8 -*-
# @Time    : 20/7/30 12:07
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : auto_augment.py
import random
import numpy as np
import scipy
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageOps
import cv2
from skimage import exposure


class AutoAugment(object):
    def __init__(self):

        self.policies = [
                ['Invert', 0.1, 7, 'Contrast', 0.2, 6],
                ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
                ['AutoContrast', 0.5, 8, 'CLAHE', 0.9, 2],
                ['Color', 0.4, 3, 'CLAHE', 0.6, 7],
                ['Sharpness', 0.3, 9, 'CLAHE', 0.7, 9],
                ['CLAHE', 0.6, 5, 'CLAHE', 0.5, 1],
                ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
                # ['Solarize', 0.5, 2, 'Invert', 0.0, 3],
                ['CLAHE', 0.2, 8, 'CLAHE', 0.6, 4],
                ['AutoContrast', 0.8, 4, 'Solarize', 0.2, 8],
                # ['Solarize', 0.4, 5, 'AutoContrast', 0.9, 3],
                # ['AutoContrast', 0.9, 2, 'Solarize', 0.8, 3],
                ['Color', 0.4, 3, 'Posterize', 0.3, 7],
        ]

    def __call__(self, img):
        img, policy_name = apply_policy(img, self.policies[random.randrange(len(self.policies))])
        return img


operations = {
    'ShearX': lambda img, magnitude: shear_x(img, magnitude),
    'ShearY': lambda img, magnitude: shear_y(img, magnitude),
    'TranslateX': lambda img, magnitude: translate_x(img, magnitude),
    'TranslateY': lambda img, magnitude: translate_y(img, magnitude),
    'Rotate': lambda img, magnitude: rotate(img, magnitude),
    'AutoContrast': lambda img, magnitude: auto_contrast(img, magnitude),
    'Invert': lambda img, magnitude: invert(img, magnitude),
    'Equalize': lambda img, magnitude: equalize(img, magnitude),
    'Solarize': lambda img, magnitude: solarize(img, magnitude),
    'Posterize': lambda img, magnitude: posterize(img, magnitude),
    'Contrast': lambda img, magnitude: contrast(img, magnitude),
    'Color': lambda img, magnitude: color(img, magnitude),
    'Brightness': lambda img, magnitude: brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: sharpness(img, magnitude),
    'Cutout': lambda img, magnitude: cutout(img, magnitude),
    'CLAHE': lambda img, magnitude: CLAHE(img, magnitude)
}


def apply_policy(img, policy):
    policy_name = ''
    if random.random() < policy[1]:
        img = operations[policy[0]](img, policy[2])
        policy_name = policy[0] + '_' + str(policy[2])
    if random.random() < policy[4]:
        img = operations[policy[3]](img, policy[5])
        policy_name = policy[3] + '_' + str(policy[5])
    return img, policy_name


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix


def shear_x(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]), 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def shear_y(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]), 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def translate_x(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-150 / 331, 150 / 331, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [0, 1,
                                  img.shape[1] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def translate_y(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-150 / 331, 150 / 331, 11)

    transform_matrix = np.array(
        [[1, 0, img.shape[0] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])],
         [0, 1, 0],
         [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def rotate(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-30, 30, 11)
    theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def auto_contrast(img, magnitude):
    img = ImageOps.autocontrast(img)
    return img


def invert(img, magnitude):
    img = ImageOps.invert(img)
    return img


def equalize(img, magnitude):
    img = ImageOps.equalize(img)
    return img


def solarize(img, magnitude):
    magnitudes = np.linspace(0, 256, 11)
    img = ImageOps.solarize(img, random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def posterize(img, magnitude):
    magnitudes = np.linspace(4, 8, 11)
    img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))))
    return img


def contrast(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def color(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def brightness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def sharpness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def cutout(org_img, magnitude=None):
    img = np.array(img)

    magnitudes = np.linspace(0, 60 / 331, 11)

    img = np.copy(org_img)
    mask_val = img.mean()

    if magnitude is None:
        mask_size = 16
    else:
        mask_size = int(round(img.shape[0] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])))
    top = np.random.randint(0 - mask_size // 2, img.shape[0] - mask_size)
    left = np.random.randint(0 - mask_size // 2, img.shape[1] - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    img[top:bottom, left:right, :].fill(mask_val)

    img = Image.fromarray(img)

    return img


class Cutout(object):
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        img = np.array(img)

        mask_val = img.mean()

        top = np.random.randint(0 - self.length // 2, img.shape[0] - self.length)
        left = np.random.randint(0 - self.length // 2, img.shape[1] - self.length)
        bottom = top + self.length
        right = left + self.length

        top = 0 if top < 0 else top
        left = 0 if left < 0 else top

        img[top:bottom, left:right, :] = mask_val

        img = Image.fromarray(img)

        return img

def CLAHE(img, magnitude=None):
    '''
    CLAHE_process
    '''
    # 将图像转换为LAB色彩空间
    img = np.array(img)

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 分离L通道
    l_channel, a_channel, b_channel = cv2.split(lab_img)

    # 对L通道进行CLAHE增强处理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel_enhanced = clahe.apply(l_channel)

    # 合并L通道和原来的a、b通道
    lab_img_enhanced = cv2.merge((l_channel_enhanced, a_channel, b_channel))

    # 将增强后的图像转回BGR色彩空间
    clahe_image = cv2.cvtColor(lab_img_enhanced, cv2.COLOR_LAB2BGR)
    #
    # # another CLAHE
    # img = np.array(img)
    # clahe_image = exposure.equalize_adapthist(img, clip_limit=0.03)

    return clahe_image
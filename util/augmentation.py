import logging
import random

import numpy as np
import torchvision.transforms.functional as tf
from PIL import Image, ImageOps, ImageEnhance

logger = logging.getLogger("Logger")


def get_augmentation(cfg_aug):
    if cfg_aug is None:
        logger.info(f'[{"DATA".center(9)}] [augmentation] No Augmentations')

        return

    if "operations" in list(cfg_aug):
        max_operations_per_instance = cfg_aug["max_operations_per_instance"]
        augment_p = cfg_aug["augment_p"]
        operations = cfg_aug["operations"]
        logger.info(
            f'[{"DATA".center(9)}] [augmentation] Using Stochastic Augmentation: Max Op {max_operations_per_instance}'
        )
    else:
        max_operations_per_instance = None
        augment_p = None
        operations = cfg_aug

    augmentations = []
    for aug_key, aug_param in operations.items():
        if aug_param:
            if type(aug_param) == dict:
                augmentations.append(key2aug[aug_key](**aug_param))
            else:
                augmentations.append(key2aug[aug_key](aug_param))
            logger.info(
                f'[{"DATA".center(9)}] [augmentation] [operation] {aug_key} [params] {aug_param}'
            )
        else:
            logger.info(
                f'[{"DATA".center(9)}] [augmentation] [operation] {aug_key} [NOT ACTIVATED]'
            )

    return Compose(augmentations, max_operations_per_instance, augment_p)


class Compose(object):
    def __init__(self, augmentations, max_operations_per_instance=None, augment_p=None):
        self.augmentations = augmentations
        self.max_operations_per_instance = (
            max_operations_per_instance if max_operations_per_instance else len(augmentations)
        )
        self.augment_p = augment_p if augment_p else 1.0
        self.PIL2Numpy = False

    def __call__(self, img, mask=None):

        augmentations = random.sample(self.augmentations, self.max_operations_per_instance)

        if mask is None:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img, mode="RGB")
                self.PIL2Numpy = True

            for a in augmentations:
                if random.random() < self.augment_p:
                    img = a(img)

            if self.PIL2Numpy:
                img = np.array(img)

            return img

        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img, mode="RGB")
                mask = Image.fromarray(mask, mode="L")
                self.PIL2Numpy = True

            if not type(mask) == dict:
                assert img.size == mask.size

            for a in augmentations:
                if random.random() < self.augment_p:
                    img, mask = a(img, mask)

            if self.PIL2Numpy:
                img, mask = np.array(img), np.array(mask, dtype=np.uint8)

            return img, mask


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return img, mask


class RandomHorizontallyFlipOnlyImg(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_TOP_BOTTOM),
                mask.transpose(Image.FLIP_TOP_BOTTOM),
            )
        return img, mask


class RandomVerticallyFlipOnlyImg(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img_np = np.array(img)
        return (
            tf.affine(
                img,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.BILINEAR,
                fillcolor=(
                    int(np.mean(img_np[..., 0])),
                    int(np.mean(img_np[..., 1])),
                    int(np.mean(img_np[..., 2])),
                ),
                shear=0.0,
            ),
            tf.affine(
                mask,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=0,
                shear=0.0,
            ),
        )


class RandomRotate90(object):
    def __init__(self, dummy):
        self.dummy = dummy

    def __call__(self, img, mask):
        # rotate_degree = random.random() * 2 * self.degree - self.degree
        rotate_degree = 90 * random.choice([0, 1, 2, 3])
        return (
            tf.rotate(img, rotate_degree, False, True, None),
            tf.rotate(mask, rotate_degree, False, True, None),
        )


class RandomRotate90OnlyImg(object):
    def __init__(self, dummy):
        self.dummy = dummy

    def __call__(self, img):
        # rotate_degree = random.random() * 2 * self.degree - self.degree
        rotate_degree = 90 * random.choice([0, 1, 2, 3])
        return tf.rotate(img, rotate_degree, False, True, None)


class Indentity(object):
    """Dummy augmentation operation that does nothing"""

    def __init__(self, magnitude):
        pass

    def __call__(self, img, mask):
        return img, mask


key2aug = {
    "hflip": RandomHorizontallyFlip,
    "hflip_onlyimg": RandomHorizontallyFlipOnlyImg,
    "vflip": RandomVerticallyFlip,
    "vflip_onlyimg": RandomVerticallyFlipOnlyImg,
    "rotate": RandomRotate,
    "rotate90": RandomRotate90,
    "rotate90_onlyimg": RandomRotate90OnlyImg,
    "identity": Indentity,
}

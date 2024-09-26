# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import math
from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target["masks"] = target["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target["masks"].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(
        float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size)
    )
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target["masks"] = (
            interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0]
            > 0.5
        )

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target["masks"] = torch.nn.functional.pad(
            target["masks"], (0, padding[0], 0, padding[1])
        )
    return padded_image, target


class ResizeDebug(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class _RandomErasingFullVertical(T.RandomErasing):

    def __init__(
        self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
    ):
        super().__init__(p, scale, ratio, value)
        # print(f"The value of p is: {p}")

    @staticmethod
    def get_params(img, scale, ratio, value=None):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w
        # print("inside vertical random erasing")
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = (
                erase_area * torch.empty(1).uniform_(1.2, 4.0).item() / (img_h**2)
            )

            # h = int(round(math.sqrt(erase_area * aspect_ratio)))
            h = img_h
            # w = int(round(math.sqrt(erase_area / aspect_ratio)))
            w = int(round(img_w * torch.empty(1).uniform_(scale[0], scale[1]).item()))
            if w >= img_w:
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):

        return self.eraser(img), target


class RandomErasingFullVertical(object):

    def __init__(self, *args, **kwargs):
        self.eraser = _RandomErasingFullVertical(*args, **kwargs)

    def __call__(self, img, target):

        return self.eraser(img), target


class _InstanceAwareRandomErasing(T.RandomErasing):
    def __init__(
        self,
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value=0,
        inplace=False,
        p_word=0.4,
        scale_ratios=(0.8, 1),
    ):
        super().__init__(p, scale, ratio, value, inplace)
        self.p_word = p_word
        self.scale_ratios = scale_ratios

    @staticmethod
    def get_params(img, scale, ratio, value=None, start_loc_j=0, end_loc_j=None):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        if end_loc_j is None:
            end_loc_j = img_w
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            h = img_h
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            try:
                j = torch.randint(start_loc_j, end_loc_j - w + 1, size=(1,)).item()
            except RuntimeError as e:
                print(e)
                print(f"start_loc_j: {start_loc_j}, end_loc_j: {end_loc_j}, w: {w}")
                continue
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def forward(self, img, target):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        boxes = target["boxes"]
        labels = target["labels"]
        # print(img.shape[-1])
        letter_locs = (boxes[:, 0] * img.shape[-1]).int()
        # print(letter_locs)
        space_locs_indices = torch.where(labels == 165)[0]
        scales = boxes[:, 2]
        letter_end_locs = ((boxes[:, 0] + boxes[:, 2]) * img.shape[-1]).int() + 1
        if isinstance(self.value, (int, float)):
            value = [float(self.value)]
        elif isinstance(self.value, str):
            value = None
        elif isinstance(self.value, (list, tuple)):
            value = [float(v) for v in self.value]
        else:
            value = self.value

        if value is not None and not (len(value) in (1, img.shape[-3])):
            raise ValueError(
                "If value is a sequence, it should have either a single value or "
                f"{img.shape[-3]} (number of input channels)"
            )
        if (
            len(space_locs_indices) > 0
        ):  # assuming if there is a space there are at least two words
            space_locs_indices_with_end = torch.cat(
                (
                    torch.tensor([0]),
                    space_locs_indices,
                    torch.tensor([len(letter_locs)]),
                )
            )
            word_locs_list = [
                letter_locs[i:j]
                for i, j in zip(
                    space_locs_indices_with_end[:-1], space_locs_indices_with_end[1:]
                )
            ]
            word_end_locs_list = [
                letter_end_locs[i:j]
                for i, j in zip(
                    space_locs_indices_with_end[:-1], space_locs_indices_with_end[1:]
                )
            ]
            scales_list = [
                scales[i:j]
                for i, j in zip(
                    space_locs_indices_with_end[:-1], space_locs_indices_with_end[1:]
                )
            ]
        else:
            word_locs_list = [letter_locs]
            word_end_locs_list = [letter_end_locs]
            scales_list = [scales]
        # print(word_locs_list)
        # letter_distances = letter_locs[1:] - letter_locs[:-1]
        # num_letters = len(letter_locs)
        # start_loc_j = letter_locs[0]
        # scale = (scales[0] * 0.8, scales[0])
        p_2_letters = 0.5
        min_len_2_letter = 6
        for word_locs, word_end_locs, scales in zip(
            word_locs_list, word_end_locs_list, scales_list
        ):

            if torch.rand(1) < self.p_word:
                # print(scales.shape)
                scale = torch.max(scales, dim=0).values
                scale = (self.scale_ratios[0] * scale, self.scale_ratios[1] * scale)
                # print(f"current word: {word_locs}")
                if (torch.rand(1) < p_2_letters) and len(word_locs) > min_len_2_letter:
                    # print("erasing 2 letters")
                    letter_to_keep = torch.randint(
                        len(word_locs) // 2 - 1, len(word_locs) // 2 + 1, (1,)
                    ).item()
                    word_locs = torch.cat(
                        (word_locs[:letter_to_keep], word_locs[letter_to_keep + 1 :])
                    )
                    x, y, h, w, v = self.get_params(
                        img,
                        scale=scale,
                        ratio=self.ratio,
                        value=value,
                        start_loc_j=word_locs[0],
                        end_loc_j=word_locs[letter_to_keep],
                    )
                    img = F.erase(img, x, y, h, w, v, self.inplace)
                    x, y, h, w, v = self.get_params(
                        img,
                        scale=scale,
                        ratio=self.ratio,
                        value=value,
                        start_loc_j=word_locs[letter_to_keep + 1],
                        end_loc_j=word_end_locs[-1],
                    )
                    img = F.erase(img, x, y, h, w, v, self.inplace)

                elif len(word_locs) > 1:
                    # print("erasing 1 letter")
                    x, y, h, w, v = self.get_params(
                        img,
                        scale=scale,
                        ratio=self.ratio,
                        value=value,
                        start_loc_j=word_locs[0],
                        end_loc_j=word_end_locs[-1],
                    )
                    img = F.erase(img, x, y, h, w, v, self.inplace)

            # if torch.rand(1) < self.p:

            #     x, y, h, w, v = self.get_params(
            #         img,
            #         scale=scale,
            #         ratio=self.ratio,
            #         value=value,
            #         start_loc_j=start_loc_j,
            #         end_loc_j = end_loc_j,
            #     )
            #     if (letter_locs[:-1] < y).all():
            #         break
            #     if labels[letter_locs > y][0] == 165:
            #         start_loc_j = letter_locs[letter_locs > y][1]
            #         scale = scales[letter_locs > y][1]
            #         continue
            #     img = F.erase(img, x, y, h, w, v, self.inplace)
            #     if (letter_locs < y + w).all():
            #         break
            #     else:  # assuming letter_locs is ordered
            #         start_loc_j = letter_locs[letter_locs > y][1]
            #         scale = (
            #             scales[letter_locs > y][1] * 0.8,
            #             scales[letter_locs > y][1],
            #         )

        return img


class InstanceAwareRandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = _InstanceAwareRandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img, target), target


class GaussianBlur(object):
    def __init__(self, kernel, sigma):
        self.kernel = kernel
        self.sigma = sigma

    def __call__(self, img, target):
        return T.GaussianBlur(self.kernel, self.sigma)(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))
import torch
import os
import pickle
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import datasets.transforms as T
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from fontTools.ttLib import TTFont
from faster_dan.Datasets.dataset_formatters.read2016_formatter import SEM_TOKENS as READ_SEM_TOKENS



current_dir = os.path.dirname(os.path.abspath(__file__))

if 'dataset' not in current_dir:
    current_dir = os.path.join(current_dir, 'dataset')
else:
    current_dir = os.path.join(current_dir, '.')

with open(os.path.join(current_dir, 'config.json'), 'r') as f:
    datasets_path = json.load(f)

with open(current_dir + "/default_charset.json", "r") as f:
    charset = json.load(f)

# Access the value of the datasets_path key
datasets_path = datasets_path['datasets_path']


class READ(Dataset):
    def __init__(self, mode, transform=transforms.ToTensor(), target_transform=None):
        """
        mode: train, valid, test
        """
        if mode == "val":
            mode = "valid"
        self.mode = mode
        self._transforms = transform
        ### load labels (text) from pickle file
        self.data = pickle.load(
            open(os.path.join(datasets_path, "READ_2016", "labels.pkl"), "rb")
        )
        # self.charset = charset
        self.charset = self.data["charset"]

        self.charset = [chr(item) for item in self.charset]

        self.img_labels = []
        self.img_idx = []
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data["ground_truth"][self.mode])

    def convert_str_to_tensor(self, text):
        ## convert sts to a list of char
        labels = []
        for c in text:
            if c == "¬":
                continue
            # c = ord(c)

            try:
                labels.append(self.charset.index(c))
            except Exception as e:
                print(c)
                print("#############")
                print(text)
                print("#############")
                print(self.charset)
                raise e
        return torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        example = self.data["ground_truth"][self.mode][idx]
        text = example["text"]

        path_image = os.path.join(
            datasets_path, "READ_2016/images", self.mode, str(example["idx"]) + ".jpeg"
        )
        image = Image.open(path_image)
        ## convert gray scale to RGB
        image = image.convert("RGB")

        labels = {}
        labels["labels"] = self.convert_str_to_tensor(text)
        labels["orig_size"] = torch.tensor(
            [image.size[1], image.size[0]], dtype=torch.int64
        )
        labels["size"] = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
        labels["img_idx"] = torch.tensor([idx], dtype=torch.int64)
        labels["idx"] = torch.tensor([idx], dtype=torch.int64)
        ## create fake boxe: tensor of size N x 4, where N is the number of boxes in the image
        dummy_boxes = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        ## repeat the dummy boxes N times
        labels["boxes"] = dummy_boxes.repeat(labels["labels"].shape[0], 1)

        image, labels = self._transforms(image, labels)

        return image, labels


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    scales = getattr(args, "data_aug_scales", scales)
    max_size = getattr(args, "data_aug_max_size", max_size)
    scales2_resize = getattr(args, "data_aug_scales2_resize", scales2_resize)
    scales2_crop = getattr(args, "data_aug_scales2_crop", scales2_crop)
    random_erasing = getattr(args, "random_erasing", False)

    # resize them
    data_aug_scale_overlap = getattr(args, "data_aug_scale_overlap", None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i * data_aug_scale_overlap) for i in scales]
        max_size = int(max_size * data_aug_scale_overlap)
        scales2_resize = [int(i * data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]

    datadict_for_print = {
        "scales": scales,
        "max_size": max_size,
        "scales2_resize": scales2_resize,
        "scales2_crop": scales2_crop,
    }

    if image_set == "train":
        if random_erasing:
            print(" INSIDE RANDOM ERASING ")
            random_erasing_transforms = [
                T.RandomErasingFullVertical(p=0.5, scale=(0.01, 0.04), ratio=(1, 3))
                for _ in range(5)
            ]
        else:
            print(" NO RANDOM ERASING ")
        if fix_size:
            return T.Compose(
                [
                    # T.RandomHorizontalFlip(),
                    T.RandomResize([(max_size, max(scales))]),
                    normalize,
                ]
            )

        if strong_aug:
            import datasets.sltransform as SLT

            return T.Compose(
                [
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=max_size),
                        T.Compose(
                            [
                                # T.RandomResize(scales2_resize),
                                # T.RandomSizeCrop(*scales2_crop),
                                T.RandomResize(scales, max_size=max_size),
                            ]
                        ),
                    ),
                    SLT.RandomSelectMulti(
                        [
                            # SLT.RandomCrop(),
                            SLT.LightingNoise(),
                            SLT.AdjustBrightness(2),
                            SLT.AdjustContrast(2),
                        ]
                    ),
                    normalize,
                ]
            )

        return T.Compose(
            [
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            # T.RandomResize(scales2_resize),
                            # T.RandomSizeCrop(*scales2_crop),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                normalize,
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                *random_erasing_transforms,
            ]
        )

    if image_set in ["val", "eval_debug", "train_reg", "test"]:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == "INFO":
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose(
                [
                    T.ResizeDebug((1280, 800)),
                    normalize,
                ]
            )

        return T.Compose(
            [
                T.RandomResize([max(scales)], max_size=max_size),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build_READ(image_set, args):
    transforms = make_coco_transforms(image_set, args=args)
    return READ(image_set, transforms)
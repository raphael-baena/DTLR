import os, sys

if __name__ == "__main__":
    sys.path.append(os.path.dirname(sys.path[0]))
import torch
import pickle
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision import transforms
import datasets.transforms as T
import random
import multiprocessing
from numpy.random import uniform, choice
from PIL import Image, ImageDraw, ImageFilter
from .generate_canva import generate_canva
from random import randint
import datasets.sltransform as SLT
import re

current_dir = os.path.dirname(os.path.abspath(__file__))

if "dataset" not in current_dir:
    current_dir = os.path.join(current_dir, "dataset")
else:
    current_dir = os.path.join(current_dir, ".")


with open(current_dir + "/dictionnary_category_ability_paths.json", "r") as f:
    dictionnary_category_ability_paths = json.load(f)
NEG_ELEMENT_BLUR_RADIUS_RANGE = (0.2, 1.6)
POS_ELEMENT_OPACITY_RANGE = {
    "drawing": (200, 255),
    "glyph": (150, 255),
    "image": (150, 255),
    "table": (200, 255),
    "text": (200, 255),
}
TEXT_COLORED_FREQ = 0.5


## padding ratio
padding_left_ratio_min = 0.02
padding_left_ratio_max = 0.1
padding_right_ratio_min = 0.02
padding_right_ratio_max = 0.1
padding_top_ratio_min = 0.02
padding_top_ratio_max = 0.2
padding_bottom_ratio_min = 0.02
padding_bottom_ratio_max = 0.2

with open(current_dir + "/default_charset.json", "r") as f:
    charset = json.load(f)
with open(current_dir + "/default_charset_without_accent.json", "r") as f:
    charset_without_accent = json.load(f)
charset_de = ["ß" if item == "Þ" else item for item in charset]

class Synthetic(Dataset):
    def __init__(
        self,
        mode,
        transform=transforms.ToTensor(),
        target_transform=None,
        language=None,
    ):
        """
        mode: train, valid, test
        """
        if mode == "val":
            mode = "valid"
        self.mode = mode
        self._transforms = transform
        self.charset = charset
        self.charset_without_accent = charset_without_accent
        self.img_labels = []
        self.transform = transform
        self.target_transform = target_transform
        self.images = []

        self.prop = 10
        self.language = language
        if self.language == "de":
            self.charset = charset_de
        if self.mode == "train":
            self.num_samples = 5000
        else:
            self.num_samples = 100
        self.create_synthetic_folder()

    def __len__(self):
        return self.num_samples * self.prop

    def create_synthetic_folder(self):
        if self.language is None:
            self.saving_path = "synthetic_images_symbols"
        elif self.language == "en":
            self.saving_path =  "synthetic_images_english"
        elif self.language == "de":
            self.saving_path = "synthetic_images_german"
        elif self.language == "fr":
            self.saving_path = "synthetic_images_french"
        self.saving_path = os.path.join(current_dir, self.saving_path)
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
        if not os.path.exists(os.path.join(self.saving_path, self.mode)):
            os.makedirs(os.path.join(self.saving_path, self.mode))  

    def generates_synthetic(self, text, idx, font_paths):
        font_path = np.random.choice(font_paths)
        font_path = current_dir + "/" + font_path
    
        image, xy, bounding_boxes = generate_textimage_with_bounding_boxes(
            text, font_path
        )
        image = generate_canva({"img": image, "position": xy})
        labels = {}
        labels["labels"] = []
        for char in text:
            if char in self.charset:
                labels["labels"].append(self.charset.index(char))
            else:
                decoded_char = char.encode().decode("unicode_escape")
                if decoded_char in self.charset:
                    labels["labels"].append(self.charset.index(decoded_char))
                else:
                    ## print unicode value of char

                    raise ValueError(f"char {char} not in charset")
        labels["labels"] = labels["labels"]
        labels["boxes"] = bounding_boxes
        labels["orig_size"] = [image.size[1], image.size[0]]
        labels["size"] = labels["orig_size"]
        labels["image_id"] = idx
        labels["idx"] = idx
        labels["font_path"] = font_path
        return image, labels

    def __getitem__(self, idx):
        idx = idx // self.prop
        image = Image.open(
                os.path.join(
                    self.saving_path, self.mode, f"{idx}.jpg")).convert("RGB")

        with open(
            os.path.join(
               self.saving_path, self.mode, f"{idx}.json"),
            "r",
        ) as f:
            labels_json = json.load(f)

        labels = {}
        labels["labels"] = torch.tensor(labels_json["labels"], dtype=torch.int64)
        labels["boxes"] = torch.tensor(labels_json["boxes"], dtype=torch.float32)  #
        labels["orig_size"] = torch.tensor(labels_json["orig_size"], dtype=torch.int64)
        labels["size"] = labels["orig_size"]
        labels["image_id"] = torch.tensor(labels_json["idx"], dtype=torch.int64)
        labels["idx"] = torch.tensor(labels_json["idx"], dtype=torch.int64)

        image, labels = self._transforms(image, labels)
        return image, labels

    def random_text(self, charset):
        ## sample 1 or 2
        if random.randint(1, 2) == 1:
            charset = self.charset
            d_fonts = sample_d_fonts("fonts_letters_with_accent_and_symbols")
            nb_words = random.randint(1, 5)
        else:
            charset = self.charset_without_accent
            d_fonts = sample_d_fonts("fonts_letters_with_accent_and_numbers")
            nb_words = random.randint(1, 30)
        text = []
        for i in range(nb_words):
            length_word = random.randint(1, 15)
            for j in range(length_word):
                text.append(random.choice(charset))
            if i < nb_words - 1:
                text.append(" ")
        text = "".join(text)
        if len(text) > 100:
            text = text[0:100]
        return text, d_fonts

    def random_text_from_wikitext(self):
        if self.language == "en":
            if "val" in self.mode:
                with open(current_dir + "/resources/text/en/val.txt") as f:
                    text_set = f.readlines()
            else:
                i = random.choice(range(1, 6))
                with open(current_dir + f"/resources/text/en/train_split_{i}.txt") as f:
                    text_set = f.readlines()
        elif self.language == "de":
            if "val" in self.mode:
                with open(current_dir + "/resources/text/de/val.txt") as f:
                    text_set = f.readlines()
            else:
                i = random.choice(range(1, 6))
                with open(current_dir + f"/resources/text/de/train_split_{i}.txt") as f:
                    text_set = f.readlines()
        elif self.language == "fr":
            if "val" in self.mode:
                with open(current_dir + "/resources/text/fr/val.txt") as f:
                    text_set = f.readlines
            else:
                i = random.choice(range(1, 6))
                with open(current_dir + f"/resources/text/fr/train_split_{i}.txt") as f:
                    text_set = f.readlines()

        for _ in range(100):
            current_text = random.choice(text_set)
            if len(current_text) < 2:
                continue
            current_text = current_text.split("\n")[:-1]
            idx_line = random.randint(0, len(current_text) - 1)
            current_text = current_text[idx_line]
            if current_text.startswith(" = "):
                continue
            current_text = re.sub(
                r""" \.| ,|" | :| ;| '|""",
                lambda match: match.group().strip(),
                current_text,
            )
            current_text = re.sub(r"\( ", "(", current_text)
            current_text = re.sub(r" \)", ")", current_text)

            current_text = re.sub(r" @-@ ", "-", current_text)
            current_text = re.sub(r" @.@ ", ".", current_text)
            # print(current_text)
            break

        if len(current_text) > 100:

            words = current_text.split()
            for _ in range(10):
                end_index = random.randint(
                    min(1, len(words) - 1), min(len(words) - 1, 20)
                )
                current_text = " ".join(words[:end_index])
                if len(current_text) > 100:
                    end_index = random.randint(
                        50, 100
                    )  # could be an issue for the language model if we are cutting a word in the end
                    current_text = current_text[0:end_index]
                if len(current_text) > 1:
                    break

        return current_text
    
    def generate_image(self, k, current_dir, mode):
        print("\r", f"Generating synthetic image {k+1}/{self.num_samples}", end="")
        while True:
            try:
                if random.randint(1, 2) == 1 and self.language is not None:
                    text = self.random_text_from_wikitext()
                    text = clean_text(text)
                    d_fonts = sample_d_fonts("fonts_letters_with_accent_and_symbols")
                else:
                    text, d_fonts = self.random_text(self.charset)
                    text = clean_text(text)
                image, labels = self.generates_synthetic(text, k, d_fonts)
                im_path = os.path.join(self.saving_path, mode, f"{k}.jpg")

                image.save(im_path)
                label_path = os.path.join( self.saving_path, mode, f"{k}.json"
                    )
                with open(label_path, "w") as f:
                        json.dump(labels, f)
                break
            except Exception as e:
                print(e)
                continue
                
                
            ## if ctrl+c is pressed, stop the generation
            except KeyboardInterrupt:
                break

    def generates_synthetic_data(self):
        ## check that folder synthetic_images_symbols exists


        pool = multiprocessing.Pool()
        results = [
            pool.apply_async(self.generate_image, args=(k, current_dir, self.mode))
            for k in range(self.num_samples)
        ]
        output = [p.get() for p in results]
        pool.close()


def clean_text(text):
    new_text = []
    for char in text:
        if char in charset:
            new_text.append(char)
        else:
            decoded_char = char.encode().decode("unicode_escape")
            if decoded_char in charset:
                new_text.append(decoded_char)
    return "".join(new_text)


def sample_d_fonts(ability):
    if random.randint(1, 2) == 1:
        category = "HANDWRITING"
    else:
        category = random.choice(["SANS_SERIF", "MONOSPACE", "SERIF", "DISPLAY"])
    return dictionnary_category_ability_paths[category][ability]


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

    if fix_size:
        return T.Compose(
            [
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ]
        )

    

    if image_set == "train":
        if random_erasing:
            random_erasing_transforms = [
                T.RandomErasingFullVertical(p=0.5, scale=(0.02, 0.05), ratio=(3, 6))
                for _ in range(5)
            ]

            return T.Compose(
                [
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=max_size),
                        T.Compose(
                            [
                                T.RandomResize(scales, max_size=max_size),
                            ]
                        ),
                    ),
                    normalize,
                    *random_erasing_transforms,
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.1, 1)),
                ]
            )
        
        else:
            return T.Compose(
                [
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=max_size),
                        T.Compose(
                            [
                                T.RandomResize(scales, max_size=max_size),
                            ]
                        ),
                    ),
                    T.GaussianBlur(kernel=(3, 3), sigma=(1, 1)),
                    normalize,
                ]
            )
    elif image_set == "val":

        return T.Compose(
            [
                T.RandomResize(scales[-1:], max_size=max_size),
                normalize,
            ]
        )



def generate_textimage_with_bounding_boxes(text, font_path):
    # Create a blank image

    # Load the font
    font_size_min = 30
    font_size_max = 50
    font_size = int(torch.randint(font_size_min, font_size_max + 1, (1,)).item())
    try:
        font = ImageFont.truetype(
            font_path, size=font_size, layout_engine=ImageFont.LAYOUT_BASIC
        )
    except:
        print(f"font_path: {font_path}")

    # bounding_box_text = font.getbbox(text) ##
    text_width, text_height = font.getsize(text)
    try:
        padding_top = int(
            float(
                torch.rand(1)
                .uniform_(padding_top_ratio_min, padding_top_ratio_max)
                .item()
            )
        )
        padding_bottom = int(
            float(
                torch.rand(1)
                .uniform_(padding_bottom_ratio_min, padding_bottom_ratio_max)
                .item()
            )
        )
        padding_left = int(
            float(
                torch.rand(1)
                .uniform_(padding_left_ratio_min, padding_left_ratio_max)
                .item()
            )
        )
        padding_right = int(
            float(
                torch.rand(1)
                .uniform_(padding_right_ratio_min, padding_right_ratio_max)
                .item()
            )
        )
    except:
        padding_top = 0
        padding_bottom = 0
        padding_left = 0
        padding_right = 0

    img_height = padding_top + padding_bottom + text_height
    img_width = padding_left + padding_right + text_width

    xy = (padding_left, padding_bottom)

    # Get bounding boxes for each character
    bounding_boxes_char = []
    new_text = ""
    for i, char in enumerate(text):
        if char != " ":
            __, __, x_max, __ = font.getbbox(text[: i + 1])
            __, __, __, y_max = font.getbbox(text[i])
            width_char, height_char = font.getmask(text[i]).size
            x_max += padding_left
            y_max += padding_bottom
            y_min = y_max - height_char
            new_text += char

        if char == " " or y_min == y_max:
            __, __, x_max, __ = font.getbbox(text[: i + 1])
            __, y_min, __, y_max = font.getbbox(text)
            width_char, __ = font.getmask(text[i]).size
            height_char = 0
            new_text += char

        x_min = x_max - width_char

        x_max = max(0, x_max)
        y_max = max(0, y_max)
        x_min = min(img_width - 1e-8, x_min)
        y_min = min(img_height - 1e-8, y_min)

        bounding_boxes_char.append([x_min, y_min, x_max, y_max])

    image = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    blur_radius = uniform(*NEG_ELEMENT_BLUR_RADIUS_RANGE)
    opacity = randint(*POS_ELEMENT_OPACITY_RANGE["text"])
    color_range = (0, 75)
    colored = choice([True, False], p=[TEXT_COLORED_FREQ, 1 - TEXT_COLORED_FREQ])
    colors = (
        tuple([randint(*color_range)] * 3)
        if not colored
        else tuple([randint(*color_range) for _ in range(3)])
    )
    colors_alpha = colors + (opacity,)

    draw.text(xy, text, font=font, fill=colors_alpha, spacing=0)
    # resize the image
    image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    image = image.resize((img_width, img_height))

    return image, xy, bounding_boxes_char


def build_synthetic_line_OCR_general(image_set, args):
    transforms = make_coco_transforms(image_set, args=args)
    # check if english is in args

    language = getattr(args, "language", None)
    if language is not None:
        print(f"USING language: {language} ")
    return Synthetic(image_set, transforms, language=language)
import os, sys

if __name__ == "__main__":
    sys.path.append(os.path.dirname(sys.path[0]))
import torch
import pickle
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import datasets.transforms as T
import random
from PIL import Image
import multiprocessing
import struct 

current_dir = os.path.dirname(os.path.abspath(__file__))

if 'dataset' not in current_dir:
    current_dir = os.path.join(current_dir, 'dataset')
else:
    current_dir = os.path.join(current_dir, '.')

with open(os.path.join(current_dir, 'config.json'), 'r') as f:
    datasets_path = json.load(f)
datasets_path = datasets_path['datasets_path']


with open(os.path.join(datasets_path, "HWDB_v1", "list_files.json"), "r") as f:
        list_of_files = json.load(f)




class Synthetic(Dataset):
    def __init__(
        self, mode, transform=transforms.ToTensor(), target_transform=None):
        """
        mode: train, valid, test
        """
        if mode == "val":
            mode = "valid"
        self.mode = mode

        self._transforms = transform
        self.charset = pickle.load(open(os.path.join(datasets_path, "HWDB_v1", "charset.pkl"), "rb"))
        self.transform = transform
        self.target_transform = target_transform
        self.prop = 10
        if mode == "train":
            self.num_samples = 5000
        else:
            self.num_samples = 500



    def __len__(self):
        return self.prop * self.num_samples
    def read_gnt_file_list(self,file_path):
        with open(file_path, 'rb') as f:
            idx = file_path.split('/')[-1].split('.')[0]
            idx = idx.split('-')[0]
            with open(file_path, 'rb') as f:
                nb_bytes = os.path.getsize(file_path)
                pos = 0
                it = 0
                list_tag = []
                list_bitmap = []
                list_width = []
                list_height = []
                list_pos = []
                while pos < nb_bytes:
                    
                    sample_size, = struct.unpack('i', f.read(4))
                    tag_code  = f.read(2)
                    tag_code = tag_code.decode('gbk')

                    label = tag_code.replace('\x00', '')

                    label = ord(label)
                    width = struct.unpack('H', f.read(2))[0]
                    height = struct.unpack('H', f.read(2))[0]
                    bitmap_data = f.read(width*height)
                    list_bitmap.append(bitmap_data)
                    list_pos.append(pos)
                    list_tag.append(label)
                    list_width.append(width)
                    list_height.append(height)
                    it += 1
                    pos += sample_size + 8
        f.closed
        return list_tag, list_width, list_height, list_pos,list_bitmap


    def generate_random_line(self):
        # mean,var = 28.040520810894904, 8.70399983799668
       
        # nb_characters = int(np.random.normal(mean, var))
        nb_characters = np.random.randint(5, 80)

        bitmaps_for_image = []
        list_characters = []
        width = 0
        height = 0
        list_widths = []
        
        for i in range(nb_characters):
            random_file = np.random.choice(list_of_files)
            list_tag, list_width, list_height, list_pos,list_bitmap = self.read_gnt_file_list(os.path.join(datasets_path, "HWDB_v1/train_raw", random_file))
            random_file_idx = np.random.choice(range(len(list_tag)))
            bitmap = list_bitmap[random_file_idx]
            bitmap = np.frombuffer(bitmap, dtype=np.uint8).reshape(list_height[random_file_idx], list_width[random_file_idx])
            width_char = list_width[random_file_idx]
            height_char = list_height[random_file_idx]
            bitmaps_for_image.append(bitmap)
            list_characters.append(list_tag[random_file_idx])
            if random.choice([True, False]):
                random_offset_x = np.random.randint(0, 20)
            else:
                random_offset_x = 0
            list_widths.append(width_char + random_offset_x)
            width += width_char + random_offset_x
            height = max(height, height_char)

        offset_right = np.random.randint(0, width//8)
        width += offset_right
        offset_left = np.random.randint(0, width//8)
        width += offset_left


        # offset_bottom = 0#np.random.randint(5, 20)
        # offset_top = 0#np.random.randint(5, 20)

        #height = #offset_bottom + offset_top
        result_array = np.ones((height,width), dtype=np.uint8) * 255
        x_offset = offset_left
        bbox = []
        for arr in bitmaps_for_image:
            if height - arr.shape[0] <= 0:
                offset_top = 0
            else:
                offset_top = np.random.randint(0, max(0,height - arr.shape[0]))
            
            result_array[offset_top:offset_top+arr.shape[0], x_offset:x_offset+arr.shape[1]] = arr
            
            left, top, right, bottom = x_offset, offset_top, x_offset+arr.shape[1], offset_top+arr.shape[0]
            x_min, y_min, x_max, y_max = left, top, right, bottom
            bbox.append((x_min, y_min, x_max, y_max))
            x_offset += list_widths.pop(0)
        return result_array, bbox, list_characters


    def __getitem__(self, idx):

        idx = idx // self.prop

        image = Image.open(os.path.join(current_dir, "synthetic_HWDB", self.mode,f"{idx}.jpg")).convert("RGB")
        with open(os.path.join(current_dir, "synthetic_HWDB", self.mode, f"{idx}.json"), "r") as f:
            labels_json = json.load(f)

        labels = {}
        labels["labels"] = torch.tensor(labels_json["labels"], dtype=torch.int64)
        labels["boxes"] = torch.tensor(labels_json["boxes"], dtype=torch.float32)  #
        labels["orig_size"] = torch.tensor(labels_json["orig_size"], dtype=torch.int64)
        labels["size"] = labels["orig_size"]
        labels["image_id"] = torch.tensor(labels_json['idx'], dtype=torch.int64)
        labels["idx"] = torch.tensor(labels_json['idx'], dtype=torch.int64)

        image, labels = self._transforms(image, labels)
        return image, labels

    def generates_synthetic(self, idx):
        result_array, bbox, list_characters = self.generate_random_line()
        image = Image.fromarray(result_array).convert("RGB")
        #upscale
        upscale = random.choice([True, False])
        if upscale:

            ratios = random.uniform(1.5, 2)
            ratios = (ratios, ratios*random.uniform(0.5, 1))
            new_size = (int(image.size[0] * ratios[0]), int(image.size[1] * ratios[1]))
            image = image.resize(new_size, Image.ANTIALIAS)
            upscaled_bbox = []
            for box in bbox:
                x_min, y_min, x_max, y_max = box
                x_min, y_min, x_max, y_max = int(x_min * ratios[0]), int(y_min * ratios[1]), int(x_max * ratios[0]), int(y_max * ratios[1])
                upscaled_bbox.append((x_min, y_min, x_max, y_max))
            bbox = upscaled_bbox
                

        labels = {}
        labels["labels"] = []
        for char in list_characters:
            labels["labels"].append(self.charset.index(char))
        
        labels["labels"] = labels["labels"]
        labels["boxes"] = bbox
        labels["orig_size"] = [image.size[1], image.size[0]]
        labels["size"] = labels["orig_size"]
        labels["image_id"] = idx
        labels["idx"] = idx
    
        return image, labels
    def generates_image(self, idx):
        print(idx)
        image, labels = self.generates_synthetic(idx)
        image.save(os.path.join(current_dir, "synthetic_HWDB", self.mode,f"{idx}.jpg"))
        with open(os.path.join(current_dir, "synthetic_HWDB", self.mode, f"{idx}.json"), "w") as f:
            json.dump(labels, f)


    def generates_synthetic_data(self):
         ## check that folder synthetic_images_symbols exists
        if not os.path.exists(os.path.join(current_dir, "synthetic_HWDB")):
            os.makedirs(os.path.join(current_dir, "synthetic_HWDB"))
        ## check that synthetic_images_symbols/self.mode exists
        if not os.path.exists(os.path.join(current_dir, "synthetic_HWDB", self.mode)):
            os.makedirs(os.path.join(current_dir, "synthetic_HWDB", self.mode))
        pool = multiprocessing.Pool() 
        results = [pool.apply_async(self.generates_image, args=(k,)) for k in range(self.num_samples)]
        output = [p.get() for p in results]
        pool.close()
 
    

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

    if fix_size:
        return T.Compose(
            [
                # T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                # augment_img,
                normalize,
            ]
        )
    if random_erasing:
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
                T.RandomErasing(p=1, scale=(0.001, 0.05), ratio=(5,6)),
                T.RandomErasing(p=1, scale=(0.001, 0.05), ratio=(5,6)),
                T.RandomErasing(p=1, scale=(0.001, 0.05), ratio=(5,6)),
                T.RandomErasing(p=1, scale=(0.001, 0.05), ratio=(5,6)),
                T.RandomErasing(p=1, scale=(0.001, 0.05), ratio=(5,6)),
                T.RandomErasing(p=1, scale=(0.001, 0.05), ratio=(5,6)),
                T.RandomErasing(p=1, scale=(0.001, 0.05), ratio=(5,6)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.05), ratio=(0.01, 0.5)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.05), ratio=(0.01, 0.5)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.05), ratio=(0.01, 0.5)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.05), ratio=(0.01, 0.5)),
                T.RandomErasing(p=0.5, scale=(0.01, 0.05), ratio=(0.01, 0.5)),
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
                normalize,
            ]
        )

def build_synthetic_HWDB(image_set, args):
    transforms = make_coco_transforms(image_set, args=args)
    return Synthetic(image_set, transforms)

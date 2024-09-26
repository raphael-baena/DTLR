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

# Access the value of the datasets_path key
datasets_path = datasets_path['datasets_path']
def treat_exponent(string):
    original_string = string
    string = string.replace("{", "")
    string = string.replace("}", "")
    underscore_in_symbol  = False
    underscore_symbole = None
    
    if "^_" in string and "^^_" not in string  and "__" not in string: ## there is no underscore symbol
        symbol = "_"
        
    elif "^_" in string and "^^_" not in string:
        symbol = "_"
        
        underscore_in_symbol = True
        underscore_symbole = string.split("_")[-1]
        if underscore_symbole == "":
            underscore_symbole ="_"
    elif "_" in string and "^_" not in string:
        underscore_in_symbol = True
        underscore_symbole = string.split("_")[-1]
        
        if underscore_symbole == "":
            underscore_symbole ="_"
        symbol =  string.split("^")
        symbol = symbol[-1].split("_")[0]
    elif "^^_" in string:
        underscore_in_symbol = True
        underscore_symbole = string.split("_")
        underscore_symbole = underscore_symbole[-1]
        if underscore_symbole == "":
            underscore_symbole ="_"
        symbol =  string.split("^")
        symbol = symbol[-1].split("_")[0]
    else:
        symbol = string.split("^")
        symbol = symbol[-1]
        if symbol == "":
            symbol ="^"


    string = string.split("^")
    result = []

    to_add = None

    if symbol == "":
        symbol ="^"
    
    for cc in string[0]:
        result.append(cc) 
    
    result.append("^" +symbol)
    if to_add is not None:
        result.append(to_add)
    to_add = None


    if underscore_in_symbol:
        result.append("_"+underscore_symbole)

    if to_add is not None:
        result.append(to_add)
    return result



        
def treat_indice(string):
    string = string.split("_")
    result = []
    question_mark_in_symbol = False
    symbol = string[-1]

    if symbol == "":
        symbol ="_"
  
    for cc in string[0]:
        result.append(cc)
    result.append("_"+ symbol)
    return result
        


class ramanacoil(Dataset):
    def __init__(self, mode, transform=transforms.ToTensor(), target_transform=None):
        """ 
        mode: train, valid, test
        """
        if mode =='val':
            mode = 'valid'
            

        self.mode = mode
        self._transforms = transform
        ### load labels (text) from pickle file
        self.data = pickle.load(open(os.path.join(datasets_path,'Ramanacoil/',"labels.pkl"), "rb"))
        np.random.seed(0)
        charset = np.random.permutation(self.data["charset"]).tolist()
        self.charset = charset
        np.random.seed()

        self.img_labels = []
        self.img_idx = []
        self.transform = transform
        self.target_transform = target_transform
        

    
    def __len__(self):
        return len(self.data["ground_truth"][self.mode])

    def convert_str_to_tensor(self, text):
    #     labels = []
    #     text = text.split(" ")
    #     for c in text:
    #         c = c.replace("?", "")
    #         if c== "<SPACE>":
    #             c = " "
    #         elif "_" in c:
    #             for cc in c:
    #                 labels.append(self.charset.index(cc))
    #         elif "," in c:
    #             ## split the string
    #             c = c.split(",")
    #             for cc in c:
    #                 if cc == "":
    #                     cc = ","
    #                 labels.append(self.charset.index(cc))
    #         elif ":^3" in c:
    #             c = c.split(":^3")
    #             for cc in c:
    #                 if cc == "":
    #                     cc = ":^3"
    #                 labels.append(self.charset.index(cc))
    #         elif ":.3" in c:
    #             c = c.split(":.3")
    #             for cc in c:
    #                 if cc == "":
    #                     cc = ":.3"
    #                 labels.append(self.charset.index(cc))
    #         elif "/" in c and "=/" not in c and ":/" not in c:
    #             c = c.split("/")[0]
    #             labels.append(self.charset.index(c))
    #         elif '^' in c:
    #             c = treat_exponent(c)
    #             for cc in c:
    #                 labels.append(self.charset.index(cc))
    #         elif "_" in c: 
    #             c = treat_indice(c)
    #             for cc in c:
    #                 labels.append(self.charset.index(cc))
    #         else:
    #             labels.append(self.charset.index(c))
        labels = []
        text = text.split(" ")
        for c in text:
            c = c.replace("?", "")
            if "/" in c:
                c = c.split("/")[0]
            if c == "<CATCHWORD":
                continue
            if "^" in c:
                c_0 = c.split("^")[0]
                s_0 = "^" + c.split("^")[-1]
                labels.append(self.charset.index(c_0))
                labels.append(self.charset.index(s_0))
            elif "__" in c and c[0].isalpha():
                c_0 = c.split("__")[0]
                s_0 = "__"
                labels.append(self.charset.index(c_0))
                labels.append(self.charset.index(s_0))

            elif "__" in c:
                for cc in c:
                    labels.append(self.charset.index(cc))
            else:
                labels.append(self.charset.index(c))          

        

        return torch.tensor(labels, dtype=torch.int64)

    
    def __getitem__(self, idx):
        example = self.data["ground_truth"][self.mode][idx]

        if example["idx"] in ["T3_0810", "T3_0052", "T3_0621", "T3_0489", "T3_1127", "T3_0607", "T3_0424", "T3_1073", "T3_1261"]:
            idx+=1
            example = self.data["ground_truth"][self.mode][idx]

        text = example["text"]
     
        path_image = os.path.join(datasets_path,"Ramanacoil",self.mode ,"img",example["idx"]+".png")
        if self.mode =="valid":
            path_image = os.path.join(datasets_path,"Ramanacoil","train","img",example["idx"]+".png")
        # check if the image exists
        if not os.path.exists(path_image):
            path_image = os.path.join(datasets_path,"Ramanacoil",self.mode ,"img",example["idx"]+".jpg")
            if self.mode =="valid":
                path_image = os.path.join(datasets_path,"Ramanacoil","train","img",example["idx"]+".jpg")
        image = Image.open(path_image)
        ## convert gray scale to RGB
        image = image.convert("RGB")
        
        labels = {}
        labels["labels"] = self.convert_str_to_tensor(text)
        labels["orig_size"]  = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
        labels["size"] = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
        labels["img_idx"] = torch.tensor([idx], dtype=torch.int64)
        labels["idx"] = torch.tensor([idx], dtype=torch.int64)
        ## create fake boxe: tensor of size N x 4, where N is the number of boxes in the image
        dummy_boxes = torch.tensor([0,0,0,0], dtype=torch.float32)
        ## repeat the dummy boxes N times
        labels["boxes"] = dummy_boxes.repeat(labels["labels"].shape[0],1)

        image, labels = self._transforms(image, labels)

        return image, labels


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # config the params for data aug
        scales = [512, 544, 576, 608, 640, 672, 704, 736, 768, 800,1333]
        max_size = 1333
        scales2_resize = [400, 500, 600]
        scales2_crop = [384, 600]
        
        # update args from config files
        scales = getattr(args, 'data_aug_scales', scales)
        max_size = getattr(args, 'data_aug_max_size', max_size)
        scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
        scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

        # resize them
        data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
        if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
            data_aug_scale_overlap = float(data_aug_scale_overlap)
            scales = [int(i*data_aug_scale_overlap) for i in scales]
            max_size = int(max_size*data_aug_scale_overlap)
            scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
            scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

        datadict_for_print = {
            'scales': scales,
            'max_size': max_size,
            'scales2_resize': scales2_resize,
            'scales2_crop': scales2_crop
        }

        if image_set == 'train':

            if fix_size:
                return T.Compose([
                    # T.RandomHorizontalFlip(),
                    T.RandomResize([(max_size, max(scales))]),
                    normalize,
                ])

            if strong_aug:
                import datasets.sltransform as SLT

                return T.Compose([
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=max_size),
                        T.Compose([
                            #T.RandomResize(scales2_resize),
                            #T.RandomSizeCrop(*scales2_crop),
                            T.RandomResize(scales, max_size=max_size),
                        ])
                    ),
                    SLT.RandomSelectMulti([
                        #SLT.RandomCrop(),
                        SLT.LightingNoise(),
                        SLT.AdjustBrightness(2),
                        SLT.AdjustContrast(2),
                    ]),
                    normalize,
                ])

            return T.Compose([

                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        #T.RandomResize(scales2_resize),
                        #T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                normalize,
            ])

        if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

            if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
                print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
                return T.Compose([
                    T.ResizeDebug((1280, 800)),
                    normalize,
                ])   

            return T.Compose([
                T.RandomResize([max(scales)], max_size=max_size),
                normalize,
            ])



        raise ValueError(f'unknown {image_set}')

def build_ramanacoil(image_set, args):
    transforms = make_coco_transforms(image_set, args=args)
    return ramanacoil(image_set, transforms)
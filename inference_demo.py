##TO CLEAN
from PIL import Image
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from util.slconfig import SLConfig
from datasets import build_dataset
from finetuning import build_model_main
from evaluation import compute_cer_impact, split_labels_into_words, process_pred_string, load_model
from util import box_ops
import editdistance
import re 
from tqdm import tqdm 

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", 
        type=str, 
        default="IAM", 
        help="aiming to get the same charset and transforms"
    )
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--new_class_embedding",  action="store_true")
    parser.add_argument("--new_label_enc",  action="store_true")
    parser.add_argument("--NMS_inference",  action="store_true")
    parser.add_argument("--metrics",  type=str, default="default")
    parser.add_argument("--unicode",  action="store_true")
    parser.add_argument("--weights", type=str, default="checkpoint.pth")
    parser.add_argument("--config", type=str, default="config/Latin_CTC.py")
    parser.add_argument("--fix_enc_out_class", action="store_true")
    parser.add_argument("--TH", type=float, default=None)
    parser.add_argument("--NMS", type=float, default=None)
    parser.add_argument(
        "--detect_path", 
        type=str, 
        default=None, 
        help="can be either img or img_folder"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default=None, 
        help="if None, directly print on console."
    )
    args = parser.parse_args()
    return args

def init_args_dataset(args):
    model_config_path = args.config

    args_dataset = SLConfig.fromfile(model_config_path)
    args_dataset.dataset_file = args.dataset
    args_dataset.device = "cuda:0"

    args_dataset.coco_path = "/comp_robot/cv_public_dataset/COCO2017/"
    args_dataset.fix_size = False
    return args_dataset

if __name__ == "__main__":
    args = get_args()

    args_dataset = init_args_dataset(args)
    dataset_val = build_dataset(image_set=args.mode, args=args_dataset)
    args_dataset.charset = dataset_val.charset
    print(len(args_dataset.charset))

    model, criterion, postprocessors = build_model_main(args_dataset)
    load_model(model, args_dataset)

    dict_char = {}

    if os.path.isdir(args.detect_path):
        img_list = [os.path.join(args.detect_path, file)
                    for file in os.listdir(args.detect_path)]
    else:
        img_list = [args.detect_path]

    TH = args.TH
    NM = args.NMS

    with torch.no_grad():
        model.eval()
        for idx, img_path in tqdm(enumerate(img_list)):
            image = Image.open(img_path)
            image = image.convert("RGB")
            labels = {}

            labels["idx"] = torch.tensor([idx], dtype=torch.int64)
            labels["labels"] = torch.tensor([0], dtype=torch.int64)

            labels["orig_size"]  = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
            labels["size"] = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
            labels["img_idx"] = torch.tensor([idx], dtype=torch.int64)
            
            dummy_boxes = torch.tensor([0,0,0,0], dtype=torch.float32)
            ## repeat the dummy boxes N times
            labels["boxes"] = dummy_boxes.repeat(1, 1)
            image, targets = dataset_val._transforms(image, labels)

            try:
                output = model.cuda()(image[None].cuda())
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print("An error occurred affecting the metrics computation")
                continue
            _, _, _, predicted_labels = compute_cer_impact(
                output, [targets], dataset_val.charset, dict_char, TH=TH, NM=NM
            )
            if args.unicode:
                pred_line = "".join([
                    chr(dataset_val.charset[int(item)]) 
                        for item in predicted_labels
                ])
            if args.save_path:
                with open(args.save_path, "w") as f:
                    f.writelines(pred_line)
            else:
                print(pred_line)

            """Hint:

            Other encoding scenarios are not implemented.
            The previous `evaluation.py` code is preserved as follows and commented.
            """
                # list_preds_str.append("".join( [chr(dataset_val.charset[int(item)]) for item in predicted_labels]) )
            # else:
            #     if args.dataset in [ "IAM", "RIMES", "READ"]:
            #         preds_str = "".join([dataset_val.charset[int(item)]for item in predicted_labels])
            #         list_preds_str.append(preds_str)
            #     else:
            #         list_preds_str.append("".join([dataset_val.charset[int(item)]for item in predicted_labels]))
            
            # if args.dataset in [ "IAM", "RIMES", "READ"]:
            #     process_pred = process_pred_string(list_preds_str[-1])

            #     #cer_it = np.mean(np.array(list_dist) / np.array(list_length_gt))
            #     pred_split = split_labels_into_words(predicted_labels, dataset_val.charset)

    

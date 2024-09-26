##TO CLEAN
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
from util import box_ops
import editdistance
import re   
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="IAM")
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
args = parser.parse_args()

model_config_path = args.config

args_dataset = SLConfig.fromfile(model_config_path)
args_dataset.dataset_file = args.dataset
args_dataset.device = "cuda:0"

args_dataset.coco_path = "/comp_robot/cv_public_dataset/COCO2017/"
args_dataset.fix_size = False

if args.NMS is not None and args.TH is not None:
    list_TH = [args.TH]
    list_NM = [args.NMS]
    args.NMS_inference = True

elif not args.NMS_inference:
    list_TH = [None]
    list_NM = [None]
else:
    list_TH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    list_NM = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def load_model(model):
    device = args_dataset.device

    if not args.new_class_embedding:
        checkpoint = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model.to(device)
        return model
    features_dim = model.class_embed[0].weight.data.shape[1]
    new_charset_size = len(args_dataset.charset)
    new_class_embed = nn.Linear(
        features_dim,
        new_charset_size,
    )
    new_decoder_class_embed = nn.Linear(
        features_dim,
        new_charset_size,
    )
    new_enc_out_class_embed = nn.Linear(
        features_dim,
        new_charset_size,
    )
    if model.dec_pred_class_embed_share:
        class_embed_layerlist = [
            new_class_embed for i in range(model.transformer.num_decoder_layers)
        ]

    new_class_embed = nn.ModuleList(class_embed_layerlist)
    model.class_embed = new_class_embed.to(device)
    model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
    if not args.fix_enc_out_class:
        model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)

    if args.new_label_enc:
        model.label_enc = nn.Embedding(len(dataset_val.charset)+1,features_dim ).to(device)
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)

@torch.no_grad()
def convert_output_to_pred(outputs, targets, charset, TH=None, NM=None):
    if args.NMS_inference: 
        output = outputs

        postprocessors['bbox'].num_select = 900
        postprocessors['bbox'].nms_iou_threshold = NM

        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
        thershold =  TH
        scores = output['scores']
        labels = output['labels']
        boxes = output['boxes']
        boxes = box_ops.box_xyxy_to_cxcywh(output["boxes"])

        select_mask = scores > thershold
        sorted_box = torch.sort(boxes[select_mask][:,0], descending=False)[1]
        
        labels = labels.long()
        items = labels[select_mask][sorted_box]

        preds = [dataset_val.charset[int(item)] for item in items]
        preds_labels = [int(item) for item in labels[select_mask][sorted_box]]
        preds = [dataset_val.charset[int(item)] for item in labels[select_mask][sorted_box]]
    else:
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        pred_logits_topk = pred_logits

        pred_boxes_topk = outputs["pred_boxes"]
        __, idx = torch.sort(pred_boxes_topk[:, :, 0])

        sorted_by_x_pred_logits = torch.gather(
            pred_logits_topk,
            1,
            idx.unsqueeze(-1).expand(-1, -1, pred_logits_topk.shape[-1]),
        )
        sorted_by_x_pred_logits = sorted_by_x_pred_logits.sigmoid()


        new_pred_logits = torch.zeros(
            (
                sorted_by_x_pred_logits.shape[0],
                sorted_by_x_pred_logits.shape[1],
                sorted_by_x_pred_logits.shape[2] + 1,
            )
        ).to(device)
        new_pred_logits[:, :, 1:] = sorted_by_x_pred_logits

        eps = 0.03 / pred_logits.shape[-1]
        mask = sorted_by_x_pred_logits.sum(-1) < 1 - eps
        new_pred_logits[:, :, 0][mask] = 1 - sorted_by_x_pred_logits[mask].sum(-1)

        mask = ~mask
        new_pred_logits[:, :, 0][mask] = eps
        new_pred_logits[:, :, 1:][mask] = (
            (1 - eps)
            * sorted_by_x_pred_logits[mask]
            / sorted_by_x_pred_logits[mask].sum(-1).unsqueeze(-1)
        )
        
        pred = new_pred_logits.max(-1)[1]
        mask = pred[0] != 0
        pred_seq = pred[0][mask]

        preds = [charset[i.item() - 1] for i in pred_seq]
        preds_labels = [i.item() - 1 for i in pred_seq]

    return preds, preds_labels  


@torch.no_grad()
def character_error_rate_with_impact(predicted_str, gt_str, dict={}):
    """
    Compute the Character Error Rate (CER) between predicted and ground truth strings.

    Args:
    predicted_str (str): The predicted string.
    gt_str (str): The ground truth string.

    Returns:
    cer (float): The Character Error Rate.
    dict (dict): The dictionary containing the impact of each character on the CER.
    """

    # Define function to calculate Levenshtein distance
    def levenshtein_distance_dict(s1, s2, dict, inversed=False):
        if len(s1) < len(s2):
            return levenshtein_distance_dict(s2, s1, dict, inversed=True)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
                c = int(c1)
                if inversed:
                    c = int(c2)
                if c1 != c2:
                    if c not in dict:
                        dict[c] = 1
                    else:
                        dict[c] += 1

            previous_row = current_row
        return previous_row[-1], dict

    # Calculate Levenshtein distance between strings
    distance, dict = levenshtein_distance_dict(predicted_str, gt_str, dict)

    # CER is the Levenshtein distance divided by the length of the ground truth string
    cer = distance / max(len(gt_str), 1)
    div = max(len(gt_str), 1)
    return cer, dict, div

@torch.no_grad()
def compute_WA(gt_str, predicted_str):
    """
    Compute the Character Error Rate (CER) between predicted and ground truth strings.

    Args:
    predicted_str (str): The predicted string.
    gt_str (str): The ground truth string.

    Returns:
    cer (float): The Character Error Rate.
    """
    # Define function to calculate Levenshtein distance
    def num_correct_pred(s1, s2):
        if len(s2) == 0:
            return 0
        correct_pred = 0
        for i, c1 in enumerate(s1):
            if i> len(s2)-1:
                break
            if s2[i] == c1:
                correct_pred+=1
        return correct_pred


    distance = num_correct_pred(gt_str, predicted_str)
    wa = distance / max(len(gt_str), 1)
    return wa
@torch.no_grad()
def compute_edit_operations(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # Deletion
                    dp[i][j - 1],  # Insertion
                    dp[i - 1][j - 1],
                )  # Substitution

    i, j = m, n
    insertion_error, deletion_error, substitution_error = 0, 0, 0

    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            substitution_error += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            deletion_error += 1
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            insertion_error += 1
            j -= 1

    insertion_error += j
    deletion_error += i

    return insertion_error, deletion_error, substitution_error


@torch.no_grad()
def compute_CR(gt_str, predicted_str, dict={}):
    insertion_error, deletion_error, substitution_error = compute_edit_operations(
        gt_str, predicted_str
    )

    return (len(gt_str) - (deletion_error + substitution_error)) / len(gt_str)


@torch.no_grad()
def character_error_rate(predicted_str, gt_str):
    """
    Compute the Character Error Rate (CER) between predicted and ground truth strings.

    Args:
    predicted_str (str): The predicted string.
    gt_str (str): The ground truth string.

    Returns:
    cer (float): The Character Error Rate.
    """

    # Define function to calculate Levenshtein distance
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    # Calculate Levenshtein distance between strings

    distance = levenshtein_distance(predicted_str, gt_str)

    # CER is the Levenshtein distance divided by the length of the ground truth string
    cer = distance / max(len(gt_str), 1)
    return cer


@torch.no_grad()
def compute_cer_impact(outputs, targets, charset, dict, TH=None, NM=None):

    cer = 0
    __, predicted_labels = convert_output_to_pred(outputs, targets, charset, TH, NM)

    if len(predicted_labels) > 0: 
        cer_it, dict, div = character_error_rate_with_impact(
            predicted_labels,
            [int(item) for item in targets[0]["labels"]],
            dict,
        )
    else:
        cer_it = 1
        div = len(targets[0]["labels"])
    cer += cer_it

    return cer, dict, div, predicted_labels


@torch.no_grad()
def word_error_rate(predicted_words, gt_words):
    """
    Compute the Word Error Rate (WER) between predicted and ground truth lists of words.

    Args:
    predicted_words (list): The predicted list of words.
    gt_words (list): The ground truth list of words.

    Returns:
    wer (float): The Word Error Rate.
    """

    # Define function to calculate Levenshtein distance
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, word1 in enumerate(s1):
            current_row = [i + 1]
            for j, word2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (word1 != word2)
                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row
        return previous_row[-1]

    # Calculate Levenshtein distance between lists of words
    distance = levenshtein_distance(predicted_words, gt_words)

    # WER is the Levenshtein distance divided by the length of the ground truth list of words
    wer = distance / max(len(gt_words), 1)
    return wer


@torch.no_grad()
def split_labels_into_words(labels, charset):
    words = []
    word = []
    for label in labels:
        if label == charset.index(" "):
            if word:
                words.append(word)
                word = []
        else:
            word.append(label)
    if word:
        words.append(word)
    return words

def process_gt_string(s):
    s = s.replace("B B C", "BBC")
    s = s.replace("I T V", "ITV")
    s = s.replace(" -", "-")
    s = s.replace("- ", "-")
    s = s.replace(" -", "-")
    s = s.replace("- ", "-")
    s = s.replace(" .", ".")
    s = s.replace(" ,", ",")
    s = s.replace(""" '""", "'")
    s = s.replace("""' """, "'")
    s = re.sub(r"(\d), (\d)", r"\1,\2", s)  # Remove space after comma between digits
    s = re.sub(r"(?<=\S)€(?=\S)", " € ", s)
    return s


def process_pred_string(s):
    s = s.replace("B B C", "BBC")
    s = s.replace("I T V", "ITV")

    s = s.replace("  ", " ")
    # s = s.replace(" )", ")")
    # s = s.replace("( ", "(")
    s = s.replace(" -", "-")
    s = s.replace("- ", "-")
    s = s.replace(" .", ".")
    s = s.replace(" ,", ",")

    s = re.sub(r"(\d), (\d)", r"\1,\2", s)  # Remove space after comma between digits
    s = s.replace(""" '""", "'")
    s = s.replace("""' """, "'")

    s = re.sub(r"(?<=\S)€(?=\S)", " € ", s)
    s = re.sub(r"(?<!\.)\.\.(?!\.)", ".", s)  # '..' to '.' if not '...'
    s = s.replace(",,", ",")  # Replace ',,' with ','

    return s


def standardize_and_evaluate(gt, pred):
    gt = process_pred_string(gt)
    pred = process_pred_string(pred)
    cer_pred = editdistance.eval(gt, pred) / len(gt)
    return cer_pred


if __name__ == "__main__":
    dataset_val = build_dataset(image_set=args.mode, args=args_dataset)
    args_dataset.charset = dataset_val.charset
    model, criterion, postprocessors = build_model_main(args_dataset)
    load_model(model)

    CER_list = []
    WER_list = []
    CER_txt = []
    AR_list = []

    dict_char = {}
    preds = []

    list_preds_str = []
    list_gt_str = []

    for TH in list_TH:
        for NM in list_NM:
            if not args.NMS_inference:
                print("Not using NMS")
            CER_list = []
            CER_txt = []
            WER_list = []
            WA_list = []
            CR_list = []

            dict_char = {}
            preds = []
            list_preds_str = []
            list_gt_str = []
            list_dist =  []
            list_length_gt = []
            with torch.no_grad():
                for i in range(len(dataset_val)):
                    model.eval()

                    image, targets = dataset_val[i]
                    try:
                        output = model.cuda()(image[None].cuda())
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print("An error occurred affecting the metrics computation")
                        continue
                    cer_it, dict_char, div, predicted_labels = compute_cer_impact(output, [targets], dataset_val.charset, dict_char, TH=TH, NM=NM)
                    if args.unicode:
                        list_preds_str.append("".join( [chr(dataset_val.charset[int(item)]) for item in predicted_labels]) )
                        list_gt_str.append("".join([chr(dataset_val.charset[int(item)]) for item in targets['labels']]))
                    else:
                        if args.dataset in [ "IAM", "RIMES", "READ"]:
                            preds_str = "".join([dataset_val.charset[int(item)]for item in predicted_labels])
                            gt_str = "".join([dataset_val.charset[int(item)]for item in targets['labels']])
                            list_preds_str.append(preds_str)
                            list_gt_str.append(gt_str)
                        else:
                            list_preds_str.append("".join([dataset_val.charset[int(item)]for item in predicted_labels]))
                            list_gt_str.append("".join([dataset_val.charset[int(item)]for item in targets['labels']]))
                    
                    dist_txt = editdistance.eval(list_gt_str[-1], list_preds_str[-1])
                    cer_txt = dist_txt / len(list_gt_str[-1])
                    if args.dataset in [ "IAM", "RIMES", "READ"]:
                        process_gt = process_pred_string(list_gt_str[-1])
                        process_pred = process_pred_string(list_preds_str[-1])
                        dist_it = editdistance.eval(process_gt, process_pred)
                        list_dist.append(dist_it)


                        list_length_gt.append(len(process_gt))
                        cer_it = sum(list_dist) / sum(list_length_gt) #DAN CER
                        #cer_it = np.mean(np.array(list_dist) / np.array(list_length_gt))
                        gt_split = split_labels_into_words([int(item) for item in targets["labels"]], dataset_val.charset)
                        pred_split = split_labels_into_words(predicted_labels, dataset_val.charset)
                        wer_it = word_error_rate(gt_split, pred_split)







                    if args.metrics == "default":
                        cer_it = cer_it 
                        wer_it = word_error_rate(
                            split_labels_into_words([int(item) for item in targets["labels"]], dataset_val.charset),
                            split_labels_into_words(predicted_labels, dataset_val.charset),
                        )
                        CER_list.append(cer_it)
                        WER_list.append(wer_it)
                        if not args.NMS_inference:
                            print("\r cer {:2f} +- {:2f}, cer txt {:2f} +- {:2f},  wer {:2f} +- {:2f}, it {}/{}".format(np.mean(CER_list),np.std(CER_list)*1.96/np.sqrt(len(CER_list)),np.mean(CER_txt),np.std(CER_txt)*1.96/np.sqrt(len(CER_txt)),np.mean(WER_list),np.std(WER_list)*1.96/np.sqrt(len(WER_list)),i,len(dataset_val)),end = '')
                        else:
                            print("\r cer {:2f} +- {:2f}, cer txt {:2f} +- {:2f},  wer {:2f} +- {:2f}, it {}/{}, TH {}, NM {}".format(np.mean(CER_list),np.std(CER_list)*1.96/np.sqrt(len(CER_list)),np.mean(CER_txt),np.std(CER_txt)*1.96/np.sqrt(len(CER_txt)),np.mean(WER_list),np.std(WER_list)*1.96/np.sqrt(len(WER_list)),i,len(dataset_val), TH, NM),end = '')
                    
                    if args.metrics == 'CER_only':
                        CER_list.append(cer_it)
                        if not args.NMS_inference:
                            print("\r cer {:2f} +- {:2f}, cer txt {:2f} +- {:2f}, it {}/{}".format(np.mean(CER_list),np.std(CER_list)*1.96/np.sqrt(len(CER_list)),np.mean(CER_txt),np.std(CER_txt)*1.96/np.sqrt(len(CER_txt)),i,len(dataset_val)),end = '')
                        else:
                            print("\r cer {:2f} +- {:2f}, cer txt {:2f} +- {:2f}, it {}/{}, TH {}, NM {}".format(np.mean(CER_list),np.std(CER_list)*1.96/np.sqrt(len(CER_list)),np.mean(CER_txt),np.std(CER_txt)*1.96/np.sqrt(len(CER_txt)),i,len(dataset_val), TH, NM),end = '')
                    if args.metrics == 'chinese':
                        cer_it = cer_it 
                        AR_it = 1-cer_it
                        CR_it = compute_CR([int(item) for item in targets["labels"]],
                        predicted_labels)
                        CER_list.append(cer_it)
                        AR_list.append(AR_it)
                        CR_list.append(CR_it)
                        if not args.NMS_inference:
                            print("\r AR {:2f} +- {:2f}, CR {:2f} +- {:2f},  cer txt {:2f} +- {:2f}, it {}/{}".format(np.mean(AR_list),np.std(AR_list)*1.96/np.sqrt(len(AR_list)),np.mean(CR_list),np.std(CR_list)*1.96/np.sqrt(len(CR_list)),np.mean(CER_txt),np.std(CER_txt)*1.96/np.sqrt(len(CER_txt)),i,len(dataset_val)),end = '')
                        else:
                            print("\r AR {:2f} +- {:2f}, CR {:2f} +- {:2f},  cer txt {:2f} +- {:2f}, it {}/{}, TH {}, NM {}".format(np.mean(AR_list),np.std(AR_list)*1.96/np.sqrt(len(AR_list)),np.mean(CR_list),np.std(CR_list)*1.96/np.sqrt(len(CR_list)),np.mean(CER_txt),np.std(CER_txt)*1.96/np.sqrt(len(CER_txt)),i,len(dataset_val), TH, NM),end = '')
                            
                    if args.metrics == 'cipher':
                        cer_it = cer_it 
                        WA_it = compute_WA([int(item) for item in targets["labels"]],predicted_labels)
                        WA_list.append(WA_it)
                        CER_list.append(cer_it)

                        if not args.NMS_inference:
                            print("\r SER {:2f} +- {:2f}, cer txt {:2f} +- {:2f}, WA {:2f} +- {:2f}, it {}/{}".format(np.mean(CER_list),np.std(CER_list)*1.96/np.sqrt(len(CER_list)),np.mean(CER_txt),np.std(CER_txt)*1.96/np.sqrt(len(CER_txt)),np.mean(WA_list),np.std(WA_list)*1.96/np.sqrt(len(WA_list)),i,len(dataset_val)),end = '')
                        else:
                            print("\r SER {:2f} +- {:2f}, cer txt {:2f} +- {:2f}, WA {:2f} +- {:2f}, it {}/{}, TH {}, NM {}".format(np.mean(CER_list),np.std(CER_list)*1.96/np.sqrt(len(CER_list)),np.mean(CER_txt),np.std(CER_txt)*1.96/np.sqrt(len(CER_txt)),np.mean(WA_list),np.std(WA_list)*1.96/np.sqrt(len(WA_list)),i,len(dataset_val), TH, NM),end = '')
           
            ordered_char = {
                k: v
                for k, v in sorted(
                    dict_char.items(), key=lambda item: item[1], reverse=True
                )
            }
            list_char = [
                dataset_val.charset[int(item)] for item in list(ordered_char.keys())
            ]
            fig, ax = plt.subplots(figsize=(16, 8))
            ##plot the impact of each character on the CER (histogram)
            plt.bar(
                range(len(ordered_char)),
                list(ordered_char.values()),
                align="center",
                width=0.5,
            )

            # Set x-axis tick labels
            plt.xticks(range(len(ordered_char)), list_char, rotation=45, fontsize=6)

            # Set labels and title
            plt.xlabel("Character Impact on CER")
            plt.ylabel("Frequency")
            plt.title("Impact of Each Character on Character Error Rate (CER)")

            # Adjust layout
            plt.tight_layout()


            stats_dir = os.path.join("stats_dect", args.dataset)
            os.makedirs(stats_dir, exist_ok=True)
            hist_file = os.path.join(
                stats_dir, "char_impact_on_cer"  + ".png"
            )
            plt.savefig(hist_file, dpi=300)

            cer_list_file = os.path.join(
                stats_dir, "cer_list"  + ".npy"
            )
            np.save(cer_list_file, CER_list)

            # Save the dict_char
            dict_char_file = os.path.join(
                stats_dir, "dict_char"  + ".json"
            )
            with open(dict_char_file, "w") as f_dict:
                json.dump(dict_char, f_dict)
            ## save list preds, gt in txt
            list_preds_file = os.path.join(
                stats_dir, "list_preds"  + ".txt"
            )
            list_gt_file = os.path.join(
                stats_dir, "list_gt"  + ".txt"
            )
            with open(list_preds_file, "w") as f_preds, open(
                list_gt_file, "w"
            ) as f_gt:
                for i in range(len(list_preds_str)):
                    f_preds.write(f"{list_preds_str[i]}\n")
                    f_gt.write(f"{list_gt_str[i]}\n")



            stats_dir = os.path.join("stats_dect", args.dataset)
            os.makedirs(stats_dir, exist_ok=True)
            stats_dir = os.path.join("stats_dect", args.dataset)
            cer_file = os.path.join(stats_dir, f"cer_TH_{TH}_NMS_{NM}.txt")

            with open(cer_file, "w") as f_cer:
                f_cer.write(
                    f"CER (TH={TH}) (NMS={NM}): {np.mean(CER_list):.4f} +- {np.std(CER_list) * 1.96 / np.sqrt(len(CER_list)):.4f}"
                )

            if not args.NMS_inference:
                break

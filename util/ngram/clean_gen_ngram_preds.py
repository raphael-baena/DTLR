import os
import pickle
import yaml
from pathlib import Path
import sys
from tqdm import tqdm
from addict import Dict
import numpy as np

# Get the directory of the current script
current_script_path = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_script_path)

# Append the parent directory to sys.path
sys.path.append(parent_dir)
from main import build_model_main
import argparse
from util.slconfig import SLConfig
from ngram_decoder.evaluation_helpers import symbols, accent_charset, weird_charset
from prediction_helpers import get_ngram_prediction
from evaluation_helpers import standardize_and_evaluate
from datasets import build_dataset
import torch.nn as nn
import torch


def get_parser():
    parser = argparse.ArgumentParser(description="Generate preds and ngram preds")

    parser.add_argument(
        "--config_path",
        type=str,
        default="ngram_decoder/IAM.yaml",
        help="Path to the ngram model config relative to root folder",
    )
    parser.add_argument("--verbose", action="store_true", help="Print predictions")
    parser.add_argument(
        "--store_predictions", action="store_true", help="Store predictions"
    )

    args = parser.parse_args()
    return args


def load_model_and_data(model_folder_path, mode, dataset_name="IAM"):
    model_config_path = f"{model_folder_path}/config_cfg.py"
    args = SLConfig.fromfile(model_config_path)
    args.CTC_training = False
    args.fix_size = False
    args.device = "cuda"
    device = args.device
    args_dataset = SLConfig.fromfile(model_config_path)
    args_dataset.device = "cuda"
    args_dataset.coco_path = "/comp_robot/cv_public_dataset/COCO2017/"
    args_dataset.fix_size = False
    args_dataset.CTC_training = False
    args_dataset.dataset_file = dataset_name

    args_dataset.mode = mode
    dataset_val = build_dataset(image_set=args_dataset.mode, args=args_dataset)
    args_dataset.charset = dataset_val.charset
    model, criterion, postprocessors = build_model_main(args)

    if dataset_name == "READ":
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
            print("inside share dec pred class embed")
            class_embed_layerlist = [
                new_class_embed for i in range(model.transformer.num_decoder_layers)
            ]
        new_class_embed = nn.ModuleList(class_embed_layerlist)
        model.class_embed = new_class_embed.to(device)
        model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
        model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
    epoch = ""
    model_checkpoint_path = f"{model_folder_path}/checkpoint{epoch}.pth"

    # model_checkpoint_path = "logs/DINO/raph_ckpts/checkpoint0021.pth"
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.eval()

    epoch = checkpoint["epoch"]

    return model, postprocessors, dataset_val, args_dataset, epoch


def get_charsets(cfg, charset=None):
    dataset_name = cfg.dataset_name
    ngram_charset = []
    if (charset is None) or (dataset_name == "IAM"):
        charset_without_accent = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "!",
            "?",
        ]
        symbols = [
            '"',
            "#",
            "$",
            "%",
            "&",
            "'",
            "(",
            ")",
            "*",
            "+",
            ",",
            "-",
            ".",
            "/",
            ":",
            ";",
            "<",
            "=",
            ">",
            "@",
            "[",
            "\\",
            "]",
            "^",
            "_",
            "`",
            "{",
            "|",
            "}",
            "~",
            " ",
        ]
        accent_charset = [
            "à",
            "á",
            "â",
            "ã",
            "ä",
            "å",
            "ā",
            "æ",
            "ç",
            "è",
            "é",
            "ê",
            "ë",
            "ì",
            "í",
            "î",
            "ï",
            "ð",
            "ñ",
            "ò",
            "ó",
            "ô",
            "õ",
            "ö",
            "ō",
            "ø",
            "ù",
            "ú",
            "û",
            "ü",
            "ý",
            "þ",
            "ÿ",
            "À",
            "Á",
            "Â",
            "Ã",
            "Ä",
            "Å",
            "Æ",
            "Ç",
            "È",
            "É",
            "Ê",
            "Ë",
            "Ì",
            "Í",
            "Î",
            "Ï",
            "Ð",
            "Ñ",
            "Ò",
            "Ó",
            "Ô",
            "Õ",
            "Ö",
            "Ø",
            "Ù",
            "Ú",
            "Û",
            "Ü",
            "Ý",
            "Þ",
            "Ÿ",
        ]
        weird_charset = ["«", "»", "—", "’", "°", "–", "œ"]
        charset = charset_without_accent + accent_charset + weird_charset + symbols
        chars_to_ignore = symbols + accent_charset + weird_charset + ["?", "!"]
        chars_to_ignore.remove("'")

    tokens_path = str(
        cfg.root / f"n-gram/data/processed_text/tokens_{dataset_name}_char.txt"
    )

    with open(tokens_path) as f:
        for line in f:
            if line.strip() == "<space>":
                # print("appending space")
                ngram_charset.append(" ")
            else:
                ngram_charset.append(line.strip())

    if dataset_name == "READ":
        # print(charset)
        charset_str = charset
        charset_digits = [char for char in charset_str if char.isdigit()]
        charset_letters = [char for char in charset_str if char.isalpha()]
        charset_symbols = [char for char in charset_str if not char.isalnum()] + ["¾"]
        chars_to_keep = charset_letters + charset_digits

        # set(charset_digits + charset_letters + charset_symbols) == set(charset_str)
        chars_to_ignore = charset_symbols
    elif dataset_name == "RIMES":
        # print(charset)
        charset_str = ngram_charset[1:]
        # charset_digits = [char for char in charset_str if char.isdigit()]
        # charset_letters = [char for char in charset_str if char.isalpha()]
        charset_symbols = [char for char in charset_str if not char.isalnum()]
        charset_symbols.remove("'")
        print(charset_symbols)

        print(set(charset_str) - set(charset_symbols))

        # # chars_to_keep = charset_letters + charset_digits
        # # set(charset_digits + charset_letters + charset_symbols) == set(charset_str)
        chars_to_ignore = charset_symbols
        # chars_to_ignore = symbols + ["«", "»", "—", "’", "°"] + ["?", "!"]

    indices_to_ignore = [ngram_charset.index(charac) for charac in chars_to_ignore]
    return indices_to_ignore, charset, ngram_charset


def get_savedir_name(cfg):
    exp_name = cfg.ngram_model_name
    if cfg.no_uppercase_words:
        exp_name += "_no_uppercase"
    if cfg.no_digits:
        exp_name += "_no_digits"
    if cfg.no_dash:
        exp_name += "_no_dash"
    exp_savedir = cfg.model_folder_path / exp_name
    return exp_savedir


def main(args):
    cfg = yaml.safe_load(open(Path(parent_dir) / args.config_path))
    cfg = Dict(cfg)
    cfg.model_folder_path = Path(cfg.model_folder_path)
    cfg.root = Path(parent_dir).parent
    print(cfg)
    symbols.remove("'")
    chars_to_ignore = symbols + accent_charset + weird_charset + ["?", "!"]

    model, postprocessors, dataset_val, args_dataset, epoch = load_model_and_data(
        cfg.model_folder_path, cfg.mode, cfg.dataset_name
    )
    indices_to_ignore, charset, ngram_charset = get_charsets(cfg, dataset_val.charset)
    exp_savedir = get_savedir_name(cfg)
    output_txt_path = exp_savedir / f"output_predictions.txt"
    cer_preds_ngram = []
    dist_preds_ngram = []
    total_div = 0
    suffix = 0
    while exp_savedir.exists():
        suffix += 1
        exp_savedir = Path(f"{str(exp_savedir)}_{suffix}")
    os.makedirs(exp_savedir)
    if cfg.num_samples is None:
        cfg.num_samples = len(dataset_val)
        print(cfg.num_samples)
    cfg_yaml_path = exp_savedir / "cfg.yaml"
    with open(cfg_yaml_path, "w") as file:
        yaml.dump(cfg, file)
    for i in tqdm(range(0, cfg.num_samples)):
        with torch.no_grad():
            image, target = dataset_val[i]
            output = model.cuda()(image[None].cuda())

            predicted_sentence = get_ngram_prediction(
                cfg, output, indices_to_ignore, charset, ngram_charset, cfg.per_word
            )
            # print(predicted_sentence)
            target_list = [charset[label] for label in target["labels"]]
            tgt_sentence = "".join(target_list)
            cer_pred_ngram, dist_pred_ngram = standardize_and_evaluate(
                tgt_sentence, predicted_sentence, args.verbose
            )
            cer_preds_ngram.append(cer_pred_ngram)
            dist_preds_ngram.append(dist_pred_ngram)
            total_div += len(tgt_sentence)
            # print(cer_pred_ngram, dist_pred_ngram)
            if args.store_predictions:
                with open(output_txt_path, "a") as file:
                    file.write(f"Target: {tgt_sentence}\n")
                    file.write(f"Ngram : {predicted_sentence}\n")
                    file.write(f"Sentence {i} CER : {cer_pred_ngram}\n")
                    file.write(f"Sentence {i} Dist: {dist_pred_ngram}\n")

    print(f"{cfg.mode} set. Mean CER pred ngram: {np.mean(np.array(cer_preds_ngram))}")
    print(
        f"{cfg.mode} set. Mean all (DAN) CER pred ngram: {np.sum(np.array(dist_preds_ngram)) / total_div}"
    )
    with open(output_txt_path, "a") as file:
        file.write(f"Mean CER pred ngram: {np.mean(np.array(cer_preds_ngram))}\n")
        file.write(
            f"Mean all (DAN) CER pred ngram: {np.sum(np.array(dist_preds_ngram)) / total_div}\n"
        )


if __name__ == "__main__":
    args = get_parser()
    main(args)

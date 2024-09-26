from pathlib import Path
import pickle
import argparse
import re
import os

# Assuming misc.py contains the punctuation_charset list
from misc import (
    write_text_to_file,
    get_chars_to_keep,
)


def repeat_char(char):
    return f"{char} {char}"


def convert_words_to_char_format(words, chars_to_keep, standard_character_level=False):
    lines = []
    for word_list in words:

        # Standard: character-level model per sentence
        if standard_character_level:

            spaced_words = []
            for word in word_list:
                spaced_word = " ".join(char for char in word)
                spaced_words.append(spaced_word)
            text = (" <space> ").join(spaced_words)
            lines.append(text)
        else:
            # character-level model per word
            for current_word in word_list:
                word = "".join(
                    [char if char in chars_to_keep else "" for char in current_word]
                )
                if len(word) == 0:
                    continue
                word = " ".join([char for char in word])
                lines.append(word)

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="IAM")
    parser.add_argument("--root", type=str, default="./")

    parser.add_argument(
        "--additional_data",
        type=str,
        default="none",
        help='"none", "wiki", or path to additional data',
    )
    args = parser.parse_args()

    root = Path(
        "/home/kallelis/PrimitiveExtraction/PrimitiveExtraction/OCR/n-gram"
    )  # FIXME: Change this to the root directory of the project
    raw_datasets_path = root / "data/raw_text"
    proc_datasets_path = root / "data/processed_text"
    dataset_name = args.dataset
    print(raw_datasets_path)
    data = pickle.load(open(raw_datasets_path / f"labels_{dataset_name}.pkl", "rb"))
    if dataset_name == "READ":
        charset = [chr(ord_num) for ord_num in data["charset"]]
        chars_to_keep = get_chars_to_keep(charset)
    else:
        charset = data["charset"]
        chars_to_keep = get_chars_to_keep(charset)
        chars_to_keep.append("'")
    text_file_names = [raw_datasets_path / f"{dataset_name}_train_text.txt"]
    output_file_path = proc_datasets_path / f"{dataset_name}_train_text_char.txt"

    if args.additional_data == "wiki":
        print(f"Using additional data from wikitext")
        text_file_names = text_file_names + [
            raw_datasets_path / f"train_split_{k}.txt" for k in range(1, 6)
        ]
    elif args.additional_data == "none":
        print(f"No additional data, using {dataset_name} training corpus only")
    else:
        print(f"Using additional data from {args.additional_data}")
        text_file_names = text_file_names + [raw_datasets_path / args.additional_data]

    space_token = "<space>"

    for text_file_name in text_file_names:
        with open(text_file_name) as f:
            words = [text.strip().split(" ") for text in f]
            lines = convert_words_to_char_format(words, chars_to_keep)
        write_text_to_file(output_file_path, lines)

    print("Generating tokens")
    ctc_token = "<ctc>"
    charset[charset.index(" ")] = space_token
    token_file_name = f"tokens_{dataset_name}_char.txt"
    token_file_path = proc_datasets_path / token_file_name
    write_text_to_file(token_file_path, [ctc_token] + charset)

    print("Generating lexicon")
    lexicon_file_name = f"lexicon_{dataset_name}_char.txt"
    lexicon_file_path = proc_datasets_path / lexicon_file_name
    lexicon_with_repetition = [repeat_char(ctc_token)] + [
        repeat_char(char) for char in charset
    ]
    write_text_to_file(lexicon_file_path, lexicon_with_repetition)


if __name__ == "__main__":
    main()

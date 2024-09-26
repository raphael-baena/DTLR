from pathlib import Path
import pickle
import argparse
import re

# Assuming misc.py contains the punctuation_charset list
from misc import punctuation_charset, clean_text, write_text_to_file, clean_READ_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="IAM")
    args = parser.parse_args()
    root = Path("./")
    raw_datasets_path = root / "data/raw_text"
    proc_datasets_path = root / "data/processed_text"
    dataset_name = args.dataset
    data = pickle.load(open(raw_datasets_path / f"labels_{dataset_name}.pkl", "rb"))

    if dataset_name == "READ":
        punctuation_pattern = "[" + re.escape("".join(punctuation_charset)) + "]"
        train_data = data["ground_truth"]["train"]
        num_examples = len(train_data)
        cleaned_texts = [
            clean_READ_text(train_data[k]["text"], punctuation_pattern)
            for k in range(num_examples)
        ]
        raw_texts = [train_data[k]["text"].strip() for k in range(num_examples)]
    else:
        punctuation_pattern = "[" + re.escape("".join(punctuation_charset)) + "]"
        cleaned_texts = [
            clean_text(e["text"], punctuation_pattern)
            for e in data["ground_truth"]["train"]
        ]
        raw_texts = [e["text"].strip() for e in data["ground_truth"]["train"]]

    output_file_name = raw_datasets_path / f"{dataset_name}_train_text.txt"
    write_text_to_file(output_file_name, raw_texts)

    output_file_path = proc_datasets_path / f"{dataset_name}_train_text_word.txt"
    write_text_to_file(output_file_path, cleaned_texts)


if __name__ == "__main__":
    main()

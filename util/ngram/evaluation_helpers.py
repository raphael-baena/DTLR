import torch
import editdistance
import re


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


@torch.no_grad()
def remove_duplicates(sequence):
    processed_output = []
    prev_char = None

    for char in sequence:
        if char != prev_char and char != 0:
            processed_output.append(char)

        prev_char = char

    return processed_output


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
def compute_cer_impact_ngram(
    predicted_str_total,
    targets,  # batch size should be 1
    ngram_charset,
    dataset_charset,
    dict,
    return_pred_comparisons=True,
):
    cer = 0

    predicted_labels_total = [
        ngram_charset.index(charac) for charac in predicted_str_total
    ]

    if len(predicted_labels_total) > 0:
        cer_it, dict, div = character_error_rate_with_impact(
            predicted_labels_total,
            [int(item) for item in targets[0]["labels"]],
            dict,
        )
    else:
        cer_it = 1
        div = len(targets[0]["labels"])
    cer += cer_it
    if return_pred_comparisons:
        pred_comparisons = f"pred : {''.join(predicted_str_total)}\n gt   : {''.join([dataset_charset[int(item)] for item in targets[0]['labels']])}"

        return cer, dict, div, predicted_labels_total, None, pred_comparisons
    else:
        return cer, dict, div, predicted_labels_total, None


def evaluate_editdistance(gt, pred_ngram, verbose=False):
    dist_pred_ngram = editdistance.eval(gt, pred_ngram)
    cer_pred_ngram = dist_pred_ngram / len(gt)
    if verbose:
        print(f"gt   : {gt}\n")
        print(f"ngram: {pred_ngram}\n")
        print(f"cer ngram {cer_pred_ngram}: \n")
        print("______________________________\n")
    return cer_pred_ngram, dist_pred_ngram


def evaluate_editdistance_comparison(gt, pred, pred_ngram):
    dist_pred = editdistance.eval(gt, pred)
    cer_pred = dist_pred / len(gt)
    dist_pred_ngram = editdistance.eval(gt, pred_ngram)
    cer_pred_ngram = dist_pred_ngram / len(gt)
    if cer_pred_ngram > cer_pred:
        print(f"gt: {gt}\n")
        print(f"pred: {pred}\n")
        print(f"pred ngram: {pred_ngram}\n")
        print(f"cer pred {cer_pred}, cer pred ngram {cer_pred_ngram}: \n")
        print("______________________________\n")
    return cer_pred, cer_pred_ngram, dist_pred, dist_pred_ngram


def standardize_and_evaluate_comparison(gt, pred, pred_ngram):
    gt = process_pred_string(gt)
    pred_ngram = process_pred_string(pred_ngram)
    return evaluate_editdistance_comparison(gt, pred_ngram)


def standardize_and_evaluate(gt, pred_ngram, verbose=False):
    gt = process_pred_string(gt)
    pred_ngram = process_pred_string(pred_ngram)
    return evaluate_editdistance(gt, pred_ngram, verbose)


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
# symbols.remove("'")

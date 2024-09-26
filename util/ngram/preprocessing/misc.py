import re
import os

def process_pred_string(s):
    s = s.replace(""" '""", "'")
    s = s.replace("""' """, "'")
    return s

def process_pred_string_(s):
    s = s.replace("  "," ")
    s = s.replace(" )", ")")
    s = s.replace("( ", "(")
    s = s.replace(" -", "-")
    s = s.replace("- ", "-")
    s = s.replace(" .", ".")
    s = s.replace(" ,", ",")
    s = re.sub(r'(\d), (\d)', r'\1,\2', s)  # Remove space after comma between digits
    s = s.replace(""" '""", "'")
    s = s.replace("""' """, "'")
    return s

def clean_text(text, punctuation_pattern):
    text = re.sub(punctuation_pattern, r" \g<0> ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def write_text_to_file(file_path, lines):
    if os.path.exists(file_path):
        print(f"Removing existing file: {file_path}")
        os.remove(file_path)
    with open(file_path, "a") as f:
        for line in lines:
            f.write(line + "\n")


def clean_READ_text(text_str, punctuation_pattern):
    text_list = list(text_str)
    new_text_list = []
    for c in text_list:
        if c == "¬":
            continue
        new_text_list.append(c)
    text = "".join(new_text_list)
    return clean_text(text, punctuation_pattern)


def get_chars_to_keep(charset_str):
    charset_digits = [char for char in charset_str if char.isdigit()]
    charset_letters = [char for char in charset_str if char.isalpha()]
    charset_symbols = [char for char in charset_str if not char.isalnum()] + ["¾"]
    chars_to_keep = charset_letters + charset_digits
    set(charset_digits + charset_letters + charset_symbols) == set(charset_str)
    return chars_to_keep


punctuation_charset = [
    "«",
    "»",
    "—",
    "°",
    "–",
    "œ",
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
]

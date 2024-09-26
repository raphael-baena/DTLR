import torch
from torchaudio.models.decoder import ctc_decoder


def get_new_pred_logits(output, multiply_pred_logits_by=1):
    device = "cuda"
    pred_logits_topk = output["pred_logits"]
    pred_boxes_topk = output["pred_boxes"]

    __, idx = torch.sort(pred_boxes_topk[:, :, 0])

    sorted_by_x_pred_logits = torch.gather(
        pred_logits_topk,
        1,
        idx.unsqueeze(-1).expand(-1, -1, pred_logits_topk.shape[-1]),
    )
    sorted_by_x_pred_logits = (
        sorted_by_x_pred_logits.sigmoid() * multiply_pred_logits_by
    )

    new_pred_logits = torch.zeros(
        (
            sorted_by_x_pred_logits.shape[0],
            sorted_by_x_pred_logits.shape[1],
            sorted_by_x_pred_logits.shape[2] + 1,
        )
    ).to(device)
    new_pred_logits[:, :, 1:] = sorted_by_x_pred_logits

    ## computes the proba of the blank token
    eps = 0.003  # / pred_logits.shape[-1] # 0.01/num_classes
    sum_pred_logits = sorted_by_x_pred_logits.sum(-1)
    mask = sorted_by_x_pred_logits.sum(-1) < 1 - eps
    new_pred_logits[:, :, 0][mask] = 1 - sorted_by_x_pred_logits[mask].sum(-1)

    mask = ~mask

    new_pred_logits[:, :, 0][mask] = eps
    new_pred_logits[:, :, 1:][mask] = (
        (1 - eps)
        * sorted_by_x_pred_logits[mask]
        / sorted_by_x_pred_logits[mask].sum(-1).unsqueeze(-1)
    )

    return new_pred_logits


def get_word_per_word_pred(new_pred_logits, ctc_decoder, indices_to_ignore, charset):
    mask = (
        new_pred_logits[0].argmax(-1)[:, None]
        == torch.tensor(indices_to_ignore, device="cuda")[None, :]
    ).any(-1)
    ind_split = torch.where((mask.cpu()))[0]
    split_indices = [-1] + ind_split.tolist() + [len(new_pred_logits[0])]

    characs = []
    model_labels = new_pred_logits[0].argmax(-1)
    for i in range(len(split_indices) - 1):
        initial_split_index = split_indices[i]
        end_split_index = split_indices[i + 1] - 1
        # TODO: add no uppercase words, no digits
        if initial_split_index < end_split_index:
            word = new_pred_logits[0][split_indices[i] + 1 : split_indices[i + 1]][
                None, :, :
            ]
            characs += ctc_decoder(word.cpu())[0][0].words

        if split_indices[i + 1] < len(new_pred_logits[0]):
            characs += charset[model_labels[split_indices[i + 1]] - 1]

    final_sentence = "".join(characs)

    return final_sentence


def get_ctc_decoder(cfg):
    dataset_name = cfg.dataset_name
    lexicon_dir_path = cfg.root / "n-gram/data/processed_text"
    ngram_model_path = str(cfg.root / f"n-gram/models/{cfg.ngram_model_name}.binary")
    lexicon_path = str(lexicon_dir_path / f"lexicon_{dataset_name}_char.txt")
    tokens_path = str(lexicon_dir_path / f"tokens_{dataset_name}_char.txt")
    ngram_ctc_decoder = ctc_decoder(
        lm=ngram_model_path,
        lexicon=lexicon_path,
        tokens=tokens_path,
        lm_weight=cfg.ngram_weight,
        blank_token="<ctc>",
        unk_word="<unk>",
        sil_token="<space>",
    )
    return ngram_ctc_decoder


@torch.no_grad()
def get_ngram_prediction(
    cfg, outputs, indices_to_ignore, charset, ngram_charset, per_word_ngram=True
):
    ngram_decoder = get_ctc_decoder(cfg)
    output = outputs
    new_pred_logits = get_new_pred_logits(output)
    if per_word_ngram and (cfg.no_uppercase_words or cfg.no_digits):
        final_sentence = get_word_per_word_pred_2(
            new_pred_logits, ngram_decoder, indices_to_ignore, ngram_charset, cfg
        )
    elif per_word_ngram:
        final_sentence = get_word_per_word_pred(
            new_pred_logits, ngram_decoder, indices_to_ignore, charset
        )
    else:
        raise NotImplementedError("no test support for full sentence n-gram for now")
        words = ngram_decoder(new_pred_logits.cpu())[0][0].words
        words = [" " if word == "<space>" else word for word in words]
        final_sentence = "".join(words)

    return final_sentence


def get_first_non_0_charac(label_list):
    for e in label_list:
        if e > 0:
            break
    return e


def get_input_split_indices(
    new_pred_logits,
    ngram_charset,
    indices_to_ignore,
    no_uppercase_words=True,
    no_digits=False,
    no_dash=True,
):

    model_labels = new_pred_logits[0].argmax(-1)
    mask = (
        model_labels[:, None] == torch.tensor(indices_to_ignore, device="cuda")[None, :]
    ).any(-1)
    ind_split = torch.where((mask.cpu()))[0]

    split_indices = [-1] + ind_split.tolist() + [len(new_pred_logits[0])]
    if not (no_uppercase_words or no_digits):
        clean_split_indices = split_indices
    else:
        clean_split_indices = []
        for i in range(len(split_indices) - 1):

            try:
                first_charac = get_first_non_0_charac(
                    model_labels[
                        split_indices[i] + 1 : split_indices[i + 1] - 1
                    ].tolist()
                )
            except UnboundLocalError as e:
                continue
            if first_charac == 0:
                continue
            if no_uppercase_words and (ngram_charset[first_charac].isupper()):
                continue

            if no_digits and (ngram_charset[first_charac].isdigit()):
                # print('uppercase', ngram_charset[first_charac], first_charac)
                continue
            elif no_dash and (
                ngram_charset.index("-")
                in model_labels[split_indices[i] + 1 : split_indices[i + 1]].tolist()
            ):
                continue

            else:
                clean_split_indices.append(split_indices[i])
        clean_split_indices.append(len(new_pred_logits[0]))
    return split_indices, clean_split_indices


def get_word_per_word_pred_2(
    new_pred_logits, ctc_decoder, indices_to_ignore, ngram_charset, cfg
):

    model_labels = new_pred_logits[0].argmax(-1)
    split_indices, clean_split_indices = get_input_split_indices(
        new_pred_logits,
        ngram_charset,
        indices_to_ignore,
        cfg.no_uppercase_words,
        cfg.no_digits,
        cfg.no_dash,
    )
    characs = []
    max_added_indices = -1

    for i in range(len(split_indices) - 1):

        initial_split_index = split_indices[i]
        end_split_index = split_indices[i + 1]

        if (initial_split_index in split_indices[1:]) and (
            initial_split_index > max_added_indices
        ):
            characs += ngram_charset[model_labels[initial_split_index]]
            max_added_indices = initial_split_index

        if (initial_split_index < end_split_index) and (
            initial_split_index in clean_split_indices
        ):
            word = new_pred_logits[0][initial_split_index + 1 : end_split_index][
                None, :, :
            ]
            characs += ctc_decoder(word.cpu())[0][0].words
            max_added_indices = max(end_split_index - 1, max_added_indices)

        else:

            word = model_labels[initial_split_index + 1 : end_split_index]
            characs += [ngram_charset[charac_label] for charac_label in word[word > 0]]
            max_added_indices = max(end_split_index - 1, max_added_indices)

        if (end_split_index in split_indices[:-1]) and (
            end_split_index > max_added_indices
        ):
            characs += ngram_charset[model_labels[end_split_index]]
            max_added_indices = end_split_index

    final_sentence = "".join(characs)
    return final_sentence

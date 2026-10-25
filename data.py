import json
import random

import torch
from torch.utils.data import Dataset

from .consts import LEDGAR_CLASS_DESCRIPTIONS, LETTERS, PAD_ID, SEED
from .prompt import prepare_ids


class JEVDataWrapper(Dataset):
    def __init__(self, data, tokenizer, is_context_json=False):
        self.data = data
        self.tokenizer = tokenizer
        self.is_context_json = is_context_json

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        if self.is_context_json:
            context = json.dumps(
                json.loads(row["state_json"]), indent=2, ensure_ascii=False
            )
        else:
            context = row["state"]
        question = row["question"]
        choices = row["options"]
        correct = choices[row["target"].index(1.0)]

        random.shuffle(choices)

        inp = prepare_ids(self.tokenizer, context, question, choices)
        tgt = choices.index(correct)
        K = len(choices)

        return inp, tgt, K


class SCIQDataWrapper(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        context = row["support"]
        question = row["question"]
        correct = row["correct_answer"]
        choices = [row[f"distractor{i}"] for i in range(1, 4)] + [correct]

        random.shuffle(choices)

        inp = prepare_ids(self.tokenizer, context, question, choices)
        tgt = choices.index(correct)
        K = len(choices)

        return inp, tgt, K


class MMLUDataWrapper(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.context = ""
        self.question_prefix = "Choose the correct answer to the question.\n"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]["train"]
        question = self.question_prefix + row["question"]
        choices = row["choices"]
        correct = choices[row["answer"]]

        random.shuffle(choices)

        inp = prepare_ids(self.tokenizer, self.context, question, choices)
        tgt = choices.index(correct)
        K = len(choices)

        return inp, tgt, K


class ANLIDataWrapper(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.context = ""
        self.options = data.features["label"].names
        self.options_desc = {
            "entailment": "The hypothesis must be true if the premise is true",
            "neutral": "The hypothesis may or may not be true given the premise",
            "contradiction": "The hypothesis cannot be true if the premise is true",
        }
        self.question_prefix = (
            "Classify the relationship between the premise and hypothesis.\n"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        premise = row["premise"]
        hypothesis = row["hypothesis"]
        question = (
            self.question_prefix
            + "Premise: "
            + premise
            + "\n"
            + "Hypothesis: "
            + hypothesis
        )
        correct = self.options[row["label"]]

        keys = self.options[:]
        random.shuffle(keys)

        if random.random() < 0.5:
            options = {k: self.options_desc[k] for k in keys}
            inp = prepare_ids(
                self.tokenizer, self.context, question, options, options_desc=True
            )
        else:
            inp = prepare_ids(self.tokenizer, self.context, question, keys)

        tgt = keys.index(correct)
        K = len(self.options)

        return inp, tgt, K


class LEDGARDataWrapper(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.all_opts = self.data.features["label"].names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.preprocess_ledgar(self.data[index])

        context = row["text"]
        question = row["question"]
        choices = row["options"]
        correct = row["label"]

        random.shuffle(choices)
        tgt = choices.index(correct)

        if random.random() < 0.5:
            choices = {k: LEDGAR_CLASS_DESCRIPTIONS[k] for k in choices}
            inp = prepare_ids(
                self.tokenizer, context, question, choices, options_desc=True
            )
        else:
            inp = prepare_ids(self.tokenizer, context, question, choices)

        K = len(choices)

        return inp, tgt, K

    def preprocess_ledgar(self, row):
        K = random.randint(3, len(LETTERS) - 1)

        label = self.all_opts[row["label"]]
        question = "Which type of contract provision is this text?"
        others = [option for option in self.all_opts if option != label]
        options = random.sample(others, k=K)
        options += [label]
        label = options[-1]

        return {
            "text": row["text"],
            "question": question,
            "options": options,
            "label": label,
        }


class CASEHOLDDataWrapper(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        label_idx = row["label"]
        context = row["context"]
        question = (
            "Which holding correctly fills the <HOLDING> placeholder in the text?"
        )
        choices = row["endings"]
        correct = choices[label_idx]

        random.shuffle(choices)

        inp = prepare_ids(self.tokenizer, context, question, choices)
        tgt = choices.index(correct)
        K = len(choices)

        return inp, tgt, K


class NLIDataWrapper(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.options = data.features["label"].names
        self.options_desc = {
            "entailment": "The contract implies the statement",
            "neutral": "The contract does not address the statement",
            "contradiction": "The contract contradicts the statement",
        }
        self.question_prefix = "Does the contract support the following statement?"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        context = row["premise"]
        hypothesis = row["hypothesis"]
        question = self.question_prefix + "\n" + "Statement: " + hypothesis
        correct = self.options[row["label"]]

        keys = self.options[:]
        random.shuffle(keys)

        if random.random() < 0.5:
            options = {k: self.options_desc[k] for k in keys}
            inp = prepare_ids(
                self.tokenizer, context, question, options, options_desc=True
            )
        else:
            inp = prepare_ids(self.tokenizer, context, question, keys)

        tgt = keys.index(correct)
        K = len(self.options)

        return inp, tgt, K


class HELLASWAGDataWrapper(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        label = int(row["label"])
        context = row["ctx"]
        question = "Which continuation is most plausible?"
        choices = row["endings"]
        correct = choices[label]

        random.shuffle(choices)

        inp = prepare_ids(self.tokenizer, context, question, choices)
        tgt = choices.index(correct)
        K = len(choices)

        return inp, tgt, K


class VITAMINDataWrapper(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.options = list(set(data["label"]))
        self.options_desc = {
            "SUPPORTS": "The evidence confirms the claim",
            "NOT ENOUGH INFO": "The evidence is not sufficient to confirm or contradict the claim",
            "REFUTES": "The evidence contradicts the claim",
        }
        self.question_prefix = "Does the evidence support the following claim?"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        context = row["evidence"]
        hypothesis = row["claim"]
        question = self.question_prefix + "\n" + "Claim: " + hypothesis
        correct = row["label"]

        keys = self.options[:]
        random.shuffle(keys)

        if random.random() < 0.5:
            options = {k: self.options_desc[k] for k in keys}
            inp = prepare_ids(
                self.tokenizer, context, question, options, options_desc=True
            )
        else:
            inp = prepare_ids(self.tokenizer, context, question, keys)

        tgt = keys.index(correct)
        K = len(self.options)

        return inp, tgt, K


def collate_fn(batch):
    tokens, labels, K = zip(*batch)
    B = len(batch)

    maxlen_T = max(t.shape[0] for t in tokens)
    tokens_padded = torch.full(size=(B, maxlen_T), fill_value=PAD_ID)
    attention_mask = torch.zeros(B, maxlen_T)

    for i, input in enumerate(tokens):
        tokens_padded[i, maxlen_T - input.shape[0] :] = input
        attention_mask[i, maxlen_T - input.shape[0] :] = 1

    labels = torch.tensor(labels, dtype=torch.long)
    K = torch.tensor(K, dtype=torch.long)

    return tokens_padded, attention_mask, labels, K


def filter_jev(ds, state_col: str, max_len: int):
    return ds.filter(
        lambda r: (
            r["kind"] == "choice"
            and len(r["options"]) <= len(LETTERS)
            and max(r["target"]) == 1.0
            and len(set(r["options"])) == len(r["options"])
            and len(r[state_col]) <= max_len
        )
    )


def ds_sample(ds, n: int, seed: int = SEED):
    return ds.shuffle(seed=seed).select(range(min(len(ds), n)))

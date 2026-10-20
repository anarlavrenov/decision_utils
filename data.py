from torch.utils.data import Dataset
import random
import json
from .prompt import prepare_ids
from consts import LETTERS

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
      context = json.dumps(json.loads(row["state_json"]), indent=2, ensure_ascii=False)
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
    self.options = ["entailment", "neutral", "contradiction"]
    self.question_prefix = "Classify the relationship between the premise and hypothesis.\n"

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    row = self.data[index]
    premise = row["premise"]
    hypothesis = row["hypothesis"]
    question = self.question_prefix + "Premise: " + premise + "\n" + "Hypothesis: " + hypothesis
    correct = self.options[row["label"]]

    options = self.options[:]

    random.shuffle(options)

    inp = prepare_ids(self.tokenizer, self.context, question, options)
    tgt = options.index(correct)
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

        inp = prepare_ids(self.tokenizer, context, question, choices)
        tgt = choices.index(correct)
        K = len(choices)

        return inp, tgt, K

    def preprocess_ledgar(self, row):
        K = random.randint(3, len(LETTERS) - 1)

        label = self.all_opts[row["label"]]
        question = "Classify the following statement."
        others = [option for option in self.all_opts if option != label]
        options = random.sample(others, k=K)
        options += [label]
        label = options[-1]

        return {
            "text": row["text"],
            "question": question,
            "options": options,
            "label": label
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
    question = "Which holding correctly fills the <HOLDING> placeholder in the text?"
    choices = row["endings"]
    correct = choices[label_idx]

    random.shuffle(choices)

    inp = prepare_ids(self.tokenizer, context, question, choices)
    tgt = choices.index(correct)
    K = len(choices)

    return inp, tgt, K

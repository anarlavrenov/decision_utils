from torch.utils.data import Dataset
import random
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
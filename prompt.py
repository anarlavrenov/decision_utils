from .consts import LETTERS


def prepare_ids(
    tokenizer,
    context: str,
    question: str,
    options: dict | list,
    options_desc: bool = False,
):
    keys = list(options)
    assert len(keys) <= len(LETTERS), f"Number of options is more than {len(LETTERS)}"

    if options_desc:
        options_text = "\n".join(
            f"{letter} = {key}: {options[key]}" for letter, key in zip(LETTERS, keys)
        )
    else:
        options_text = "\n".join(
            f"{letter} = {opt}" for letter, opt in zip(LETTERS, options)
        )

    messages = [
        {
            "role": "user",
            "content": (
                f"Context: {context}\n\n"
                f"Question: {question}\n\n"
                f"Options: \n{options_text}\n\n"
                "Answer with a single letter."
            ),
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )["input_ids"][0]

    return input_ids

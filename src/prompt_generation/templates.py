def get_input_length(text, tokenizer) -> int:
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_length = input_ids.shape[1]
    return input_length


def create_chemuref_prompt(
    dev_example: dict, train_example1: dict, train_example2: dict
) -> str:

    # add in context examples
    prompt = ""
    for inContext in [train_example1, train_example2]:
        prompt += "Context: %s\nQuestion: What does %s contain?\nAnswer: %s\n\n" % (
            inContext["context"],
            inContext["anaphor"],
            " | ".join(inContext["gold_antecedents"]),
        )

    # add the example itself
    prompt += "Context: %s\nQuestion: What does %s contain?\nAnswer:" % (
        dev_example["context"],
        dev_example["anaphor"],
    )

    return prompt, len(prompt)


def create_chemuref_prompt_without_answer(example: dict) -> str:
    return "Context: %s\nQuestion: What does %s contain?\nAnswer:" % (
        example["context"],
        example["anaphor"],
    )


def create_chemuref_prompt_with_answers(example: dict) -> str:
    return "Context: %s\nQuestion: What does %s contain?\nAnswer: %s\n" % (
        example["context"],
        example["anaphor"],
        " | ".join(example["gold_antecedents"]),
    )

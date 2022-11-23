import json

def read_jsonl(filepath: str) -> dict:
    data = dict()
    with open(filepath, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            data[example["example_id"]] = example
    return data
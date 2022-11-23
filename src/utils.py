import json


def read_jsonl(filepath: str) -> dict:
    """NOTE: This is not generalizable"""
    data = dict()
    with open(filepath, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            data[example["example_id"]] = example
    return data


def write_jsonl(data: list, filepath: str) -> None:

    with open(filepath, "w") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")


def read_txt(filepath: str) -> list:
    data = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def write_txt(data: list, filepath: str) -> None:
    with open(filepath, "w") as f:
        for line in data:
            f.write(line + "\n")


def read_json(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def write_json(d: dict, filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(d, f, indent=4)

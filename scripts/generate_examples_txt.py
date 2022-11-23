import json
import os
import sys

from utils import read_jsonl, write_txt


def main():

    test_filepath = sys.argv[1]
    exp_dir = sys.argv[2]
    test_data = read_jsonl(test_filepath)
    test_ids = list(test_data.keys())
    example_filepath = os.path.join(exp_dir, "test_examples.txt")
    write_txt(test_ids, example_filepath)


if __name__ == "__main__":
    main()

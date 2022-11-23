import os
import sys
import json


def read_json(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def read_jsonl(filepath: str) -> dict:
    data = dict()
    with open(filepath, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            data[example["example_id"]] = example
    return data


def write_jsonl(data: dict, filepath: str):

    with open(filepath, "w") as f:
        for _, example in data.items():
            f.write(json.dumps(example) + "\n")


def add_false_positives(data: dict, fp: dict):

    for docID, fps in fp.items():

        # first find the last exampleID
        example_num = 0 if docID not in ["0952"] else 1
        example_id = "%s-0%s" % (docID, example_num)
        while example_id in data:
            example_num += 1
            example_id = (
                "%s-0%s" % (docID, example_num)
                if example_num < 10
                else "%s-%s" % (docID, example_num)
            )
        last_example_id = (
            "%s-0%s" % (docID, example_num - 1)
            if example_num - 1 < 10
            else "%s-%s" % (docID, example_num - 1)
        )
        last_example = data[last_example_id]

        # start adding the false positives
        for [anaphor, index] in fps:
            new_context = last_example["context"][: index + len(anaphor)]
            data[example_id] = {
                "example_id": example_id,
                "context": new_context,
                "anaphor": anaphor,
                "gold_antecedents": [],
                "preceding_anaphors": [],
            }

            # updating example_id
            example_num += 1
            example_id = (
                "%s-0%s" % (docID, example_num)
                if example_num < 10
                else "%s-%s" % (docID, example_num)
            )

    return data


def delete_false_negatives(data: dict, fn: dict):

    fp_ids = []
    for docID, fns in fn.items():

        for [anaphor, index] in fns:

            for example_id, example in data.items():

                if docID in example_id:

                    if (
                        example["anaphor"] == anaphor
                        and example["context"][index : index + len(anaphor)]
                        == example["context"][-1 - len(anaphor) : -1]
                    ):
                        fp_ids.append(example_id)
                        break

    for example_id in fp_ids:
        del data[example_id]

    return data


def main():

    test_fp_filepath = "/srv/share5/nghia6/codebases/mixture/chemuref_anaphor_detection/test_false_positives.json"
    test_fn_filepath = "/srv/share5/nghia6/codebases/mixture/chemuref_anaphor_detection/test_false_negatives.json"
    test_filepath = "/srv/share5/nghia6/data/ChemuRef_v2/lm_trimmed_data/test.jsonl"
    test_predicted_filepath = "/srv/share5/nghia6/data/ChemuRef_v2/lm_trimmed_data/test_predicted_anaphors.jsonl"
    testsmall_fp_filepath = "/srv/share5/nghia6/codebases/mixture/chemuref_anaphor_detection/test_small_false_positives.json"
    testsmall_fn_filepath = "/srv/share5/nghia6/codebases/mixture/chemuref_anaphor_detection/test_small_false_negatives.json"
    testsmall_filepath = (
        "/srv/share5/nghia6/data/ChemuRef_v2/lm_trimmed_data/test_small.jsonl"
    )
    testsmall_predicted_filepath = "/srv/share5/nghia6/data/ChemuRef_v2/lm_trimmed_data/test_small_predicted_anaphors.jsonl"

    for (fn, fp, data, predicted_filepath) in [
        (
            read_json(testsmall_fn_filepath),
            read_json(testsmall_fp_filepath),
            read_jsonl(testsmall_filepath),
            testsmall_predicted_filepath,
        ),
        (
            read_json(test_fn_filepath),
            read_json(test_fp_filepath),
            read_jsonl(test_filepath),
            test_predicted_filepath,
        ),
    ]:

        # add false positives first
        data = add_false_positives(data, fp)

        # then delete false negatives
        data = delete_false_negatives(data, fn)

        # output predicted datasets
        write_jsonl(data, predicted_filepath)


if __name__ == "__main__":
    main()

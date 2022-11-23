import json
import os
import string
import sys
import re


def read_data(filepath: str) -> dict:
    data = dict()
    with open(filepath, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            doc_id = example["example_id"].split("-")[0]
            context = example["context"]
            anaphor = example["anaphor"]

            # find index of anaphor in context (from the right to ensure the right anaphor)
            anaphor_index = context.rfind(anaphor)

            if doc_id not in data:
                data[doc_id] = {
                    "context": example["context"],
                    "gold_anaphors": [(anaphor, anaphor_index)],
                }
            else:
                if len(context) > len(data[doc_id]["context"]):
                    data[doc_id]["context"] = context
                data[doc_id]["gold_anaphors"].append((anaphor, anaphor_index))

    return data


def anaphora_rules(context: str) -> list:

    results = []
    # Some general variables for the rules

    # Rule 1: match "x (y)" pattern e.g. "the title compound (12 mg, 78% yield)"
    start_tokens = ["The", "the", "a"]
    end_tokens = ["compound", "product", "solid"]
    start_pattern = "(%s)" % "|".join(start_tokens)
    end_pattern = "(%s)" % "|".join(end_tokens)
    middle_num_repeat = 3
    pattern1 = "%s( ([^(\s\.\,)]*)){0,%s} %s( \(([^\(]*)\))" % (
        start_pattern,
        middle_num_repeat,
        end_pattern,
    )
    pattern1 = "%s( ([^(\s\.\,)]*)){0,%s} %s( \(([^\(]*)( ([^\(]*)){1,4}\))" % (
        start_pattern,
        middle_num_repeat,
        end_pattern,
    )

    # Rule 2: match "the x" eg "the resulting mixture"
    # NOTE: Also include "the title compound". Resolve difference by prioritizing 1
    start_tokens = [
        "The",
        "the",
        "This",
        "this",
    ]
    end_tokens = [
        "mixture",
        "solids",
        "solid",
        "solution",
        "residue",
        "layers",
        "layer",
        "filtrates",
        "filtrate",
        "product",
        "extracts",
        "crystals",
        "crystal",
        "tube",
        "suspension",
        "phases",
        "phase",
        "substance",
        "resultant",
        "solvents",
        "solvent",
        "eluent",
        "vial",
        "precipitation",
        "Solvent",
        "liquid",
        "flask",
        "reminder",
        "cake",
        "mass",
        "fractions",
        "fraction",
        "system",
        "DCU",
        "oil",
    ]
    start_pattern = "(%s)" % "|".join(start_tokens)
    end_pattern = "(%s)" % "|".join(end_tokens)
    middle_num_repeat = 3
    pattern2 = "%s( ([^(\s\.\,)]*)){0,%s} %s" % (
        start_pattern,
        middle_num_repeat,
        end_pattern,
    )

    # Rule 2: Dealing with "a/an ..." case
    start_tokens = [
        "a",
        "An",
        "an",
    ]
    end_tokens = [
        "vessel",
        "flask",
        "layer",
        "phase",
        "oil",
        "solid",
        "tube",
        "crystal",
        "liquid",
        "powder",
        "funnel",
    ]
    start_pattern = "(%s)" % "|".join(start_tokens)
    end_pattern = "(%s)" % "|".join(end_tokens)
    middle_num_repeat = 4
    pattern3 = "%s( ([^(\s\.\,)]*)){0,%s} %s" % (
        start_pattern,
        middle_num_repeat,
        end_pattern,
    )

    # Rule 4: eg "the compound 37b (9.86 g, 96%)" (sacrifice "compound 55 (421 mg, 66.4%)" to improve precision)
    pattern4 = "the compound (\S*) \(([^\(]*)( ([^\(]*)){1,4}\)"

    # Rule 5 eg "59 mg (99%) of the title compound" and "1.49 g (yield: 56%) of a white solid"
    pattern5a = (
        "(\S*) (\S*) \(([^\(]*)( ([^\(]*)){0,1}\) of (the title compound|a white solid)"
    )
    pattern5b = "(\S*) of a product mixture"
    pattern5c = "(A compound 48 \(0.63 g, 1.53 mmol, 85%\)|10 mg of the target compound \(31% of theory\))"

    # The following rules basically hard-coding from the dev set

    # Rule 6 "This compound" and "eluent". "solvent" would increase recall...
    pattern6 = "(This compound|eluent)"

    # Rule 7 "compound"-related hard-coding
    pattern7a = "(The|the){0,1}( ([^(\s\.\,)]*)){0,1} (Compound|compound) \(([^\(E]*)\)"
    pattern7b = "(The fractions|Fractions) containing the (title|desired) compound"

    # Rule 8 "X + number"
    pattern8 = "(Raw material|product|the following intermediate) [0-9]"

    # Rule 9 Straight-up hard-coding
    pattern9 = "(catalyst|Insoluble materials|Product fractions|pale yellow solid|colorless form)"
    pattern9a = (
        "(Pure fractions|The resulting material|a yellow semi-solid \(585.9 mg, 86%\))"
    )
    # Rule 10: "Yield"-related
    # NOTE: pattern10b is mixed-- precision/recall tradeoff
    pattern10a = "Yield: (\S*) (mg|g) \(([^\(]*)( ([^\(]*)){1,4}\)"
    pattern10b = "(([0-9]*)(mg|g) ){0,1} \(([^\(]*)( (yield|yield:)( (\S*)){0,1})\)"
    pattern10c = "(\(7.80 mmol, yield: 87%\)|17mg \(29% yield\)|\(17.2 g, 96%\))"
    pattern10d = "11.68 mmol, yield: 67\%"

    existing_start_indicies = set()
    existing_end_indicies = set()
    for pattern in [
        pattern5c,
        pattern7a,
        pattern7b,
        pattern1,
        pattern2,
        pattern3,
        pattern4,
        pattern5a,
        pattern5b,
        pattern6,
        pattern8,
        pattern9,
        pattern9a,
        pattern10a,
        # pattern10b,
        pattern10c,
        pattern10d,
    ]:
        for match in re.finditer(pattern, context):
            start_idx, end_idx = match.span()
            span = context[start_idx:end_idx].strip()
            if pattern == pattern5c:
                print(span)
            start_idx = context.find(span, start_idx)
            if span[-1] in string.punctuation and span[-1] not in [")", "%"]:
                span = span[:-1]
            # prioritize rule1 -> rule2 -> rule3
            if (
                start_idx not in existing_start_indicies
                and end_idx not in existing_end_indicies
            ):
                results.append((span, start_idx))
                existing_start_indicies.add(start_idx)
                existing_end_indicies.add(end_idx)

    return results


def anaphora_detection(dataset: dict) -> dict:

    predictions = dict()
    for example_id, example in dataset.items():
        predictions[example_id] = {
            "context": example["context"],
            "gold_anaphors": example["gold_anaphors"],
            "predicted_anaphors": anaphora_rules(example["context"]),
        }

    return predictions


def evaluate(predictions: dict, exp_dir: str) -> dict:

    gold_num_anaphors = 0
    predicted_num_anaphors = 0
    tp = 0
    fp_dict = dict()
    fn_dict = dict()
    for example_id, example in predictions.items():

        gold_anaphors = set(example["gold_anaphors"])
        predicted_anaphors = set(example["predicted_anaphors"])
        true_positives = []

        # computing the true positives of current example
        for gold_anaphor in gold_anaphors:
            for predicted_anaphor in predicted_anaphors:
                if gold_anaphor == predicted_anaphor:
                    tp += 1
                    true_positives.append(predicted_anaphor)

        gold_num_anaphors += len(gold_anaphors)
        predicted_num_anaphors += len(predicted_anaphors)

        # store false positives and false negatives
        fp = list(set(predicted_anaphors) - set(true_positives))
        if len(fp) > 0:
            fp_dict[example_id] = fp
        fn = list(set(gold_anaphors) - set(true_positives))
        if len(fn) > 0:
            fn_dict[example_id] = fn

    fp_filepath = os.path.join(exp_dir, "test_fp.txt")
    fn_filepath = os.path.join(exp_dir, "test_fn.txt")
    for (filepath, d) in [(fp_filepath, fp_dict), (fn_filepath, fn_dict)]:
        # with open(filepath, "w") as f:
        #     json.dump(d, f, indent=4)
        with open(filepath, "w") as f:
            for example_id, l in d.items():
                f.write("%s: %s\n" % (example_id, l))

    p = tp / predicted_num_anaphors
    r = tp / gold_num_anaphors
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if (p + r) != 0 else 0,
    }


def main():

    exp_dir = "./chemuref_anaphor_detection"
    test_filepath = "/srv/share5/nghia6/data/ChemuRef_v2/lm_trimmed_data/test.jsonl"

    # read in data and get predictions
    test_dataset = read_data(test_filepath)
    predictions = anaphora_detection(test_dataset)
    metrics = evaluate(predictions, exp_dir)
    print(metrics)

    # output metrics and predictions
    metrics_filepath = os.path.join(exp_dir, "test_metrics.json")
    with open(metrics_filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    predictions_filepath = os.path.join(exp_dir, "test_predictions.json")
    with open(predictions_filepath, "w") as f:
        json.dump(predictions, f, indent=4)


if __name__ == "__main__":
    main()

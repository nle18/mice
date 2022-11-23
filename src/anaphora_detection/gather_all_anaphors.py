"""Gather all anaphors and partition them accordingly (eg train/dev/test)

Outputs: 
train/
    - k04trial0.txt 
    - k04trial1.txt 
    - ...
    - k04total.txt 
    - ...
    - kfewshottotal.txt
    - kfulltotal.txt 
dev.txt 
test.txt 

Each file contains lines which are 
anaphor anaphor_count 

Sorted by anaphor_count 
"""
from collections import defaultdict
import json
import os
import sys


def output_start_end_tokens(anaphor_counts, total_count, out_filepath):

    # gather start and end counts
    start_counts, end_counts = defaultdict(int), defaultdict(int)
    for anaphor, count in anaphor_counts.items():
        start, end = anaphor.split(" ")[0], anaphor.split(" ")[-1]
        start_counts[start] += count
        end_counts[end] += count

    # output
    sorted_start_counts = sorted(start_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_end_counts = sorted(end_counts.items(), key=lambda x: x[1], reverse=True)
    start_filepath = out_filepath[:-4] + "_start.txt"
    end_filepath = out_filepath[:-4] + "_end.txt"
    for (sorted_counts, filepath) in [
        (sorted_start_counts, start_filepath),
        (sorted_end_counts, end_filepath),
    ]:
        with open(filepath, "w") as f:
            for token, count in sorted_counts:
                f.write("%s %s %0.4f\n" % (token, count, count / total_count))


def output_useful_statistics(anaphor_counts, total_count, out_filepath):

    # coarse-grained num count
    categories = {
        "num_count%s1" % c: {"num_anaphors": 0, "counts": 0} for c in ["=", ">"]
    }
    for anaphor, count in anaphor_counts.items():
        key = "num_count=1" if count == 1 else "num_count>1"
        categories[key]["num_anaphors"] += 1
        categories[key]["counts"] += count
    categories["num_count=1"]["probs"] = (
        categories["num_count=1"]["counts"] / total_count
    )
    categories["num_count>1"]["probs"] = (
        categories["num_count>1"]["counts"] / total_count
    )

    # distribution of span length
    lengths = {"length=%s" % l: {"num_anaphors": 0, "counts": 0} for l in range(1, 11)}
    lengths["length>10"] = {"num_anaphors": 0, "counts": 0}
    for anaphor, count in anaphor_counts.items():
        anaphor_len = len(anaphor.split(" "))
        if anaphor_len > 10:
            lengths["length>10"]["num_anaphors"] += 1
            lengths["length>10"]["counts"] += count
        else:
            lengths["length=%s" % anaphor_len]["num_anaphors"] += 1
            lengths["length=%s" % anaphor_len]["counts"] += count
    for l in range(1, 11):
        lengths["length=%s" % l]["probs"] = (
            lengths["length=%s" % l]["counts"] / total_count
        )
    lengths["length>10"]["probs"] = lengths["length>10"]["counts"] / total_count

    # output statistics
    stats = {"num_counts": categories, "anaphor_lengths": lengths}
    info_filepath = out_filepath.replace("txt", "json")
    with open(info_filepath, "w") as f:
        json.dump(stats, f, indent=4)


def gather_anaphor(in_filepath: str, out_filepath: str) -> dict:

    # read in and gather anaphor
    anaphor_counts = defaultdict(int)
    with open(in_filepath, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            anaphor_counts[example["anaphor"]] += 1

    # output sorted count corresponding file
    sorted_counts = sorted(anaphor_counts.items(), key=lambda x: x[1], reverse=True)
    total_count = sum(list(map(lambda x: x[1], sorted_counts)))
    with open(out_filepath, "w") as f:
        for anaphor, count in sorted_counts:
            f.write("%s %s %0.4f\n" % (anaphor, count, count / total_count))

    # output categorization information
    output_start_end_tokens(anaphor_counts, total_count, out_filepath)
    output_useful_statistics(anaphor_counts, total_count, out_filepath)
    return anaphor_counts


def main():

    # exp_dir = "./chemuref_anaphor_detection/anaphor_counts"
    exp_dir = "./arrau_anaphor_detection/anaphor_counts"
    os.makedirs(exp_dir, exist_ok=True)
    dataset_dir = "/srv/share5/nghia6/data/ChemuRef/lm_trimmed_data/"
    dataset_dir = "/srv/share5/nghia6/data/ARRAU_RST/"

    # first handle training data
    train_counts_dir = os.path.join(exp_dir, "train")
    os.makedirs(train_counts_dir, exist_ok=True)
    # all_counts = defaultdict(int)
    # for k in ["k04", "k08", "k16", "k32", "k64"]:
    #     k_counts = defaultdict(int)
    #     for trial in range(5):
    #         in_filepath = os.path.join(dataset_dir, "train", k, "trial%s.jsonl" % trial)
    #         out_filepath = os.path.join(train_counts_dir, "%strial%s.txt" % (k, trial))
    #         anaphor_counts = gather_anaphor(in_filepath, out_filepath)

    #         # total anaphor for each k and all few-shot data
    #         for anaphor, count in anaphor_counts.items():
    #             k_counts[anaphor] += count
    #             all_counts[anaphor] += count

    #     # generate anaphors for each k
    #     out_filepath = os.path.join(train_counts_dir, "%s.txt" % k)
    #     sorted_counts = sorted(k_counts.items(), key=lambda x: x[1], reverse=True)
    #     total_count = sum(list(map(lambda x: x[1], sorted_counts)))
    #     with open(out_filepath, "w") as f:
    #         for anaphor, count in sorted_counts:
    #             f.write("%s %s %0.4f\n" % (anaphor, count, count / total_count))
    #     output_start_end_tokens(k_counts, total_count, out_filepath)
    #     output_useful_statistics(k_counts, total_count, out_filepath)

    # # generate all few-shot anaphors
    # out_filepath = os.path.join(train_counts_dir, "kfewshot.txt")
    # sorted_counts = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
    # total_count = sum(list(map(lambda x: x[1], sorted_counts)))
    # with open(out_filepath, "w") as f:
    #     for anaphor, count in sorted_counts:
    #         f.write("%s %s %0.4f\n" % (anaphor, count, count / total_count))
    # output_start_end_tokens(all_counts, total_count, out_filepath)
    # output_useful_statistics(all_counts, total_count, out_filepath)

    # generate kfull anaphors
    in_filepath = os.path.join(dataset_dir, "train", "kfull.jsonl")
    out_filepath = os.path.join(train_counts_dir, "kfull.txt")
    gather_anaphor(in_filepath, out_filepath)

    # then handle dev and test
    for set_type in ["test", "dev"]:
        in_filepath = os.path.join(dataset_dir, "%s.jsonl" % set_type)
        out_filepath = os.path.join(exp_dir, "%s.txt" % set_type)
        gather_anaphor(in_filepath, out_filepath)


if __name__ == "__main__":
    main()

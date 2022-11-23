import json

def main():
    input_data_path = "data/unlabeled/uspto_5000_paras.json"
    output_data_path = "data/unlabeled/uspto_5000_paras_chemref.jsonl"

    # Read in data
    output_list = []
    with open(input_data_path, "r") as f:
        unlabeled_data = json.load(f)
        # print(len(unlabeled_data))
        
        for doc_id, sen_list in unlabeled_data.items():
            # print(doc_id)
            # print(' '.join(sen_list[1:]))

            example = {}
            example["example_id"] = doc_id

            # Remove the title may work better
            example["context"] = ' '.join(sen_list[1:])
            # example["context"] = ' '.join(sen_list[:])
            
            example["anaphor"] = ''
            example["gold_antecedents"] = []
            example["preceding_anaphors"] = []
            output_list.append(example)

            # break
        # print("done")

    # Write out data
    with open(output_data_path, "w") as f:
        for example in output_list:
            print(example["example_id"])
            json.dump(example, f)
            f.write("\n")

if __name__ == "__main__":
    main()
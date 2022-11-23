import json

def main():
    input_data_path = "data/unlabeled/uspto_5000_paras_anaphor_pred.json"
    output_data_path = "data/unlabeled/uspto_5000_paras_anaphor_pred_chemref_0_2000.jsonl"

    # Read in data
    output_list = []
    with open(input_data_path, "r") as f:
        anaphora_pred = json.load(f)
        print(len(anaphora_pred))
        
    # Convert to chemref format
    for doc_id in anaphora_pred:
        # print(doc_id)
        # print(anaphora_pred[doc_id])

        predicted_anaphors = anaphora_pred[doc_id]["predicted_anaphors"]
        context = anaphora_pred[doc_id]["context"]
        preceding_anaphors = []
        # print(predicted_anaphors)

        if len(predicted_anaphors) > 8:
            continue

        # print(len(predicted_anaphors))

        for anaphor_idx, (predicted_anaphor, start_idx) in enumerate(predicted_anaphors):
            
            # print(context[start_idx:start_idx+len(predicted_anaphor)])
            end_idx = start_idx + len(predicted_anaphor)
            assert predicted_anaphor == context[start_idx:end_idx]

            context_truncated = context[:end_idx]
            context_truncated = context_truncated if context_truncated.endswith(".") else context_truncated + "."

            output_list.append({
                "example_id": doc_id+f"-{anaphor_idx:02d}", 
                "context": context_truncated, 
                "anaphor": predicted_anaphor,
                "gold_antecedents": [],
                "preceding_anaphors": [],
            })

            preceding_anaphors.append(predicted_anaphor)

    # Write to file
    print(len(output_list))
    with open(output_data_path, "w") as f:
        for example in output_list[:2000]:
            json.dump(example, f)
            f.write("\n")


if __name__ == "__main__":
    main()
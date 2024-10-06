import json
import sys
import argparse

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def calculate_accuracy(path_pred, path_gt):

    data_pred = read_jsonl(path_pred)
    data_gt = read_jsonl(path_gt)

    questionID_gt = {item['question_id']: item['answers'] for item in data_gt}
    choices = ['A', 'B']
    
    correct_pairs = 0
    total_pairs = 0
    for i in range(0, len(data_pred), 2):  # QA paris
        if i + 1 < len(data_pred) and data_pred[i]['question_id'] in questionID_gt and data_pred[i + 1]['question_id'] in questionID_gt:
            assert data_pred[i]['text'].upper() in choices, f"Invalid answer choice: {data_pred[i]['question_id']}, {data_pred[i]['text'].upper()}"
            assert data_pred[i+1]['text'].upper() in choices, f"Invalid answer choice: {data_pred[i+1]['question_id']}, {data_pred[i+1]['text'].upper()}"
            answer_correct = data_pred[i]['text'].upper() == questionID_gt[data_pred[i]['question_id']] and \
                             data_pred[i+1]['text'].upper() == questionID_gt[data_pred[i + 1]['question_id']]
            if answer_correct:
                correct_pairs += 1
            total_pairs += 1

    accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_pred", type=str, default='./playground/data/eval/mm-vet/llava-mm-vet.jsonl')
    parser.add_argument("--path_gt", type=str, default='./playground/data_VT/eval_images_gen_vt/mmvet_gen_vt/VTGenerator-13B/merge.jsonl')
    args = parser.parse_args()

    accuracy = calculate_accuracy(args.path_pred, args.path_gt)
    print(f"path_pred: {args.path_pred}")
    print(f"path_gt: {args.path_gt}")
    print(f"Accuracy: {accuracy}")
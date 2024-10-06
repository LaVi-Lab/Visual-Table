import json
import sys
import argparse

def dataset_with_vt(dataset_path, dataset_vt_path, dataset_with_vt_path):
    visual_table_prompt_pre = "Visual table:"
    visual_table_prompt_post = "Based on the given image and given visual table, answer the following question:"

    to_print = True

    dict_image_vt = {}
    with open(dataset_vt_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            if to_print:
                print(json_data)
                to_print = False
            question_id = json_data['question_id']
            text = json_data['text']
            dict_image_vt[question_id] = text

    to_print = True

    with open(dataset_path, 'r') as file, open(dataset_with_vt_path, 'w') as output_file:
        for line in file:
            json_data = json.loads(line)
            image = json_data['image']
            assert image in dict_image_vt, f"image {image} not in dict_image_vt"
            json_data['text'] = visual_table_prompt_pre + '\n' + \
                                dict_image_vt[image] + '\n' + \
                                visual_table_prompt_post + '\n' + \
                                json_data['text']
            if to_print:
                print(json_data)
                to_print = False
            output_file.write(json.dumps(json_data) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='./playground/data/eval/mm-vet/llava-mm-vet.jsonl')
    parser.add_argument("--dataset_vt_path", type=str, default='./playground/data_VT/eval_images_gen_vt/mmvet_gen_vt/VTGenerator-13B/merge.jsonl')
    parser.add_argument("--dataset_with_vt_path", type=str, default='./playground/data_VT/eval/mm-vet/mmvet_with_VTGenerator-13B_gen_vt.jsonl')
    args = parser.parse_args()

    print(f"dataset_path = {args.dataset_path}")
    print(f"dataset_vt_path = {args.dataset_vt_path}")
    print(f"dataset_with_vt_path = {args.dataset_with_vt_path}")
    dataset_with_vt(args.dataset_path, args.dataset_vt_path, args.dataset_with_vt_path)
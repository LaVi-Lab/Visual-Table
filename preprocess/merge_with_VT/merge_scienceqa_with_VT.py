import json
import sys
import argparse

def dataset_with_gen_vt(dataset_path, dataset_vt_path, dataset_with_vt_path):
    visual_table_prompt_pre = "Visual table:"
    visual_table_prompt_post = "Based on the given image and given visual table, answer the following question:"

    dict_image_vt = {}
    with open(dataset_vt_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            question_id = json_data['question_id']
            text = json_data['text']
            dict_image_vt[question_id] = text

    data_with_gen_vt = []
    with open(dataset_path, 'r') as file:
        data = json.load(file)
        for line in data:
            #   {
            #     "id": "4",
            #     "conversations": [
            #     {
            #         "from": "human",
            #         "value": "Which figure of speech is used in this text?\nSing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans.\n\u2014Homer, The Iliad\nA. chiasmus\nB. apostrophe"
            #     },
            #     {
            #         "from": "gpt",
            #         "value": "B"
            #     }
            #     ]
            # },
            # {
            #     "id": "5",
            #     "image": "5/image.png",
            #     "conversations": [
            #     {
            #         "from": "human",
            #         "value": "<image>\nContext: People can use the engineering-design process to develop solutions to problems. One step in the process is testing if a potential solution meets the requirements of the design.\nThe passage below describes how the engineering-design process was used to test a solution to a problem. Read the passage. Then answer the question below.\n\nGordon was an aerospace engineer who was developing a parachute for a spacecraft that would land on Mars. He needed to add a vent at the center of the parachute so the spacecraft would land smoothly. However, the spacecraft would have to travel at a high speed before landing. If the vent was too big or too small, the parachute might swing wildly at this speed. The movement could damage the spacecraft.\nSo, to help decide how big the vent should be, Gordon put a parachute with a 1 m vent in a wind tunnel. The wind tunnel made it seem like the parachute was moving at 200 km per hour. He observed the parachute to see how much it swung.\nFigure: a spacecraft's parachute in a wind tunnel.\nWhich of the following could Gordon's test show?\nA. if the spacecraft was damaged when using a parachute with a 1 m vent going 200 km per hour\nB. how steady a parachute with a 1 m vent was at 200 km per hour\nC. whether a parachute with a 1 m vent would swing too much at 400 km per hour"
            #     },
            #     {
            #         "from": "gpt",
            #         "value": "B"
            #     }
            #     ]
            # },
            if 'image' not in line: 
                data_with_gen_vt.append(line)
                continue
            
            image = line['image']
            assert image in dict_image_vt, f"image {image} not in dict_image_vt"
            line["conversations"][0]["value"] = line["conversations"][0]["value"].replace("<image>\n", "")
            line["conversations"][0]["value"] = "<image>\n" + visual_table_prompt_pre + '\n' + \
                                dict_image_vt[image] + '\n' + \
                                visual_table_prompt_post + '\n' + \
                                line["conversations"][0]["value"]
            data_with_gen_vt.append(line)
    
    with open(dataset_with_vt_path, 'w') as file:
        json.dump(data_with_gen_vt, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='./playground/data/eval/scienceqa/llava_test.jsonl')
    parser.add_argument("--dataset_vt_path", type=str, default='./playground/data_VT/eval_images_gen_vt/scienceqa_gen_vt/VTGenerator-13B/merge.jsonl')
    parser.add_argument("--dataset_with_vt_path", type=str, default='./playground/data_VT/eval/scienceqa/scienceqa_with_VTGenerator-13B_gen_vt.jsonl')
    args = parser.parse_args()
    print(f"dataset_path = {args.dataset_path}")
    print(f"dataset_vt_path = {args.dataset_vt_path}")
    print(f"dataset_with_vt_path = {args.dataset_with_vt_path}")
    dataset_with_gen_vt(args.dataset_path, args.dataset_vt_path, args.dataset_with_vt_path)
import json
import random
import os
import copy
from tqdm import tqdm
import argparse

random.seed(42)

def main(args):
    vt_prompt_pre = "Visual table:"
    vt_prompt_post = "Based on the given image and given visual table, answer the following question:"

    # <image> <vt_prompt_pre> <vt> <vt_prompt_post> <question>

    vts = {}
    imageID_set = set()

    pre_path = './playground/data_VT/train_images_gen_vt/'
    paths = [
        'llava_instruct_mix665k_coco_gen_vt/VTGenerator-13B/merge.jsonl',
        'llava_instruct_mix665k_vg_gen_vt/VTGenerator-13B/merge.jsonl',
        'llava_instruct_mix665k_textcap_gen_vt/VTGenerator-13B/merge.jsonl',
        'llava_instruct_mix665k_ocrvqa_gen_vt/VTGenerator-13B/merge.jsonl'
    ]

    for path in tqdm(paths):
        with open(pre_path + path, 'r') as file:
            dataset_name = path.split('_')[3] # coco, vg, textcap, ocrvqa
            for i in file:
                line = json.loads(i)
                # {'question_id': '000000033471.jpg', 
                # 'prompt': 'Based on the given image, generate the visual table follow the following JSON format: {"scene description": "", "objects": [{"object category": "", "attribute description": "", "knowledge description": ""}, ......, {"object category": "", "attribute description": "", "knowledge description": ""}]}', 
                # 'text': '{"scene description": "This is an urban street scene during the daytime featuring a bus with advertising on its rear parked by the sidewalk. Pedestrians are visible on the sidewalk, suggesting a busy area.", "objects": [{"object category": "Bus", "attribute description": "White bus with red and black advertising graphics, text, and URLs on the rear, parked on the street", "knowledge description": "A bus is a large motor vehicle designed to carry passengers. The advertising on the bus indicates it may be used for promotional purposes in addition to public transportation."}, {"object category": "Pedestrians", "attribute description": "Two individuals standing on the sidewalk, one appears to be a woman holding a child", "knowledge description": "Pedestrians are people traveling on foot. They are using the sidewalk, which is designed for pedestrian traffic alongside a road."}, {"object category": "Sidewalk", "attribute description": "Gray concrete pathway beside the road for pedestrian use", "knowledge description": "Sidewalks provide a safe area for pedestrians to walk, separated from the vehicular traffic on the road."}, {"object category": "Buildings", "attribute description": "Partially visible structures in the background, likely commercial or residential", "knowledge description": "Buildings serve various purposes such as housing, commerce, or services and are common in urban environments."}, {"object category": "Street", "attribute description": "Asphalt road where the bus is parked, with visible lane markings", "knowledge description": "Streets are pathways for vehicles and are essential for urban transportation and connectivity."}, {"object category": "Utility Pole", "attribute description": "Tall pole with wires, likely for electrical or telecommunications purposes", "knowledge description": "Utility poles are used to support overhead power lines and various other public utilities, such as electrical cable, fiber optic cable, and related equipment like transformers."}]}', 'answer_id': 'LPwHjiVtoCLXr7yzeXsXAT', 'model_id': 'llava13b_mix199k_gen_vt_61k_epoch3', 'metadata': {}}
                assert '.jpg' in line['question_id']
                imageID = line['question_id'].replace('.jpg', '')
                assert imageID not in imageID_set
                imageID_set.add(imageID)
                item = {
                    "dataset_name": dataset_name,
                    "result": line['text']
                }
                vts[imageID] = item
                # {"000000033471": {"dataset_name": "coco", "result":"..."}

    print(len(vts))
    print(len(imageID_set))
    # 291684
    # 291684

    with open(args.gen_VT_path, 'w') as file:
        json.dump(vts, file)
        
    mix665k_with_gen_vt = []
    with open(args.llava_instruct_mix665k_path, 'r') as file:
        data = json.load(file)
        print(len(data))
        # 665298
        
        for line in tqdm(data):
            if 'image' not in line:
                mix665k_with_gen_vt.append(line)
                continue
                        
            imageID = line['image'].split('/')[-1].replace('.jpg', '')
            if "\n<image>" in line['conversations'][0]["value"]:
                line['conversations'][0]["value"] = "<image>\n" + line['conversations'][0]["value"].replace("\n<image>", "")
                        
            assert os.path.exists(os.path.join(args.image_path, line['image'])), f"{os.path.join(args.image_path, line['image'])} not exist"
            
            # Example:
            # './playground/data/coco/train2017/000000033471.jpg'
            # './playground/data/vg/VG_100K/4.jpg' or './playground/data/vg/VG_100K_2/51.jpg'
            # "./playground/data/gqa/images/2354786.jpg"
            # './playground/data/ocr_vqa/images/140031996X.jpg'
            # './playground/data/textvqa/train_images/011e7e629fb9ae7b.jpg'
            
            line['conversations'][0]["value"] = "<image>\n" + vt_prompt_pre + '\n' \
                + vts[imageID]["result"] + '\n' \
                + vt_prompt_post + '\n' \
                + line['conversations'][0]["value"].replace("<image>\n", "")
            mix665k_with_gen_vt.append(line)
            
    print(mix665k_with_gen_vt[0])
    # {'id': '000000033471', 'image': 'coco/train2017/000000033471.jpg', 'conversations': [{'from': 'human', 'value': '<image>\nVisual table:\n{"scene description": "This is an urban street scene during the daytime featuring a bus parked by the sidewalk with pedestrians nearby.", "objects": [{"object category": "Bus", "attribute description": "Large, red and white, advertising graphics on the back, parked", "knowledge description": "A bus is a large motor vehicle designed to carry passengers. It is used for public transport in urban and suburban areas."}, {"object category": "Pedestrian", "attribute description": "Adult, standing on the sidewalk, casual clothing", "knowledge description": "Pedestrians are people traveling on foot, which is a common sight in urban areas and they must follow certain safety rules when near roadways."}, {"object category": "Sidewalk", "attribute description": "Gray, concrete, beside the road", "knowledge description": "Sidewalks are pathways alongside the road designed for pedestrian use to ensure their safety by separating them from vehicular traffic."}, {"object category": "Building", "attribute description": "Partially visible in the background, appears to be a commercial establishment", "knowledge description": "Buildings in urban areas often house businesses and are part of the commercial landscape of a city."}, {"object category": "Street", "attribute description": "Asphalt, with visible lane markings, vehicles parked", "knowledge description": "Streets are public thoroughfares in a city or town, paved for vehicular traffic and often include markings for safety and regulation of traffic."}, {"object category": "Sky", "attribute description": "Partly cloudy, daylight", "knowledge description": "The sky is the expanse of air over the Earth\'s surface, often observed for weather conditions which can affect outdoor activities and mood."}, {"object category": "Electricity poles and wires", "attribute description": "Vertical poles with horizontal wires, above the street", "knowledge description": "Electricity poles and wires are infrastructure for power distribution in urban areas, essential for providing electricity to surrounding buildings and street lights."}]}\nBased on the given image and given visual table, answer the following question:\nWhat are the colors of the bus in the image?'}, {'from': 'gpt', 'value': 'The bus in the image is white and red.'}, {'from': 'human', 'value': 'What feature can be seen on the back of the bus?'}, {'from': 'gpt', 'value': 'The back of the bus features an advertisement.'}, {'from': 'human', 'value': 'Is the bus driving down the street or pulled off to the side?'}, {'from': 'gpt', 'value': 'The bus is driving down the street, which is crowded with people and other vehicles.'}]}

    with open(args.llava_instruct_mix665k_with_VT_path, 'w') as file:
        json.dump(mix665k_with_gen_vt, file)

    print(len(mix665k_with_gen_vt))
    # 665298

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_VT_path", type=str, default='./playground/data_VT/train_images_gen_vt/VTGenerator-13B_VT_292k.json')
    parser.add_argument("--llava_instruct_mix665k_path", type=str, default='/path/to/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json')
    parser.add_argument("--image_path", type=str, default='./playground/data')
    parser.add_argument("--llava_instruct_mix665k_with_VT_path", type=str, default='./playground/data_VT/train_LLaVA-VT/llava_instruct_mix665k_with_VT.json')
    args = parser.parse_args()
    main(args)

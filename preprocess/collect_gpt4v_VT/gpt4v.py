# --------------------------------------------------------
# Collect responses from GPT-4V
# Written by:
#   Yiwu Zhong
# --------------------------------------------------------
import os
import base64
import requests
from io import BytesIO
import json
import random
import time
from multiprocessing.pool import Pool

# Get OpenAI API Key from environment variable
api_key = ""  # os.environ["OPENAI_API_KEY"]
api_org = ""
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    "OpenAI-Organization": f"{api_org}"
}

system_prompt = '''
You are an AI visual assistant that can analyze a single image. Given an image, you need to perform the task of scene description. And then, you need to identify each object in the image. For each object, you need to perform 3 tasks: object category recognition, attribute description generation, and knowledge description generation.

Scene description:
1. Based on the given image, please provide a short and concise description for the scene in the image, such as the location, the time of the day (e.g., morning, evening), the event, and so on.

Object category recognition:
1. Based on the given image, please recognize the category for each object in the scene. 
2. Please cover as many objects as possible. The objects should cover not only the salient objects, but also the other objects such as the small objects, the objects in the background, the objects that are partially occluded, and so on.

Attribute description generation:
1. Based on the given image, please generate the visual attributes for each object. 
2. Visual attributes characterize the objects in images. They can be OCR characters on the object, spatial relations to surrounding objects, action relations to surrounding objects, relative size compared to surrounding objects, color, geometry shape, material, texture pattern, scene environment, motion or dynamics of objects, and so on.
3. Specially, if possible, the visual attributes could be the emotions (e.g., surprised, angry), age (e.g., young, elderly), and so on.

Knowledge description generation:
1. Based on the given image, please describe the knowledge for each object.
2. The knowledge includes object affordance, commonsense knowledge, background knowledge, and so on. 
3. Object affordance is defined as the functions supported by the objects. For example, what the objects can be used for? Note that the affordance might be altered case by case, due to deformed shape, unreliable materials, and so on.
4. Commonsense knowledge is defined as basic understandings and assumptions about daily life, human behavior, and the natural world. It also includes understanding social norms, basic cause-and-effect relationships, and simple reasoning about daily situations.
5. Background knowledge is defined as the knowledge of named entities, such as celebrities, ceremonies, festivals, and so on.

Output format:
The output content should follow the following JSON format. {"scene description": "", "objects": [{"object category": "", "attribute description": "", "knowledge description": ""}, ......, {"object category": "", "attribute description": "", "knowledge description": ""}]}. Directly output the JSON without any other content. The output MUST follow JSON format.
'''    

unavailable_list = ['000000457286', '000000104943', '000000375325', '000000152389', '000000438060', '000000149320', 
                    '000000559464', '000000280307', '000000252799', '000000280228', '000000566704', '000000077680',
                    '000000085483', '000000301077', '000000125567', '000000475696',
                    'validation_Basic_Medical_Science_20', 'validation_Clinical_Medicine_12', 
                    'oven_05062612.JPEG', 'oven_05062618.JPEG', 'oven_05062646.JPEG', 'oven_05052958.JPEG']

coco_anno_file = './image_cap_bbox_dict.json' # this file is obtained by running ./coco_preprocess.py
with open(coco_anno_file, 'r') as f:
    coco_annos = json.load(f)

def combine_pool_files(output_dir, output_json):
    # Combine all the processed files into one
    combined_contents = {}
    for file_name in os.listdir(output_dir):
        if 'combined' in file_name:
            continue
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(output_json, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All collection completed!")

# Function to encode the image
def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_inputs(message, image, detail):
    base64_image = image

    payload = {
        "model": "gpt-4-vision-preview",
        # "response_format": {"type": "json_object"},  # Enable JSON mode
        "messages": [
        {
            "role": "system",
            "content": [
                system_prompt
            ]
        }, 
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": message, 
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": detail,
                }
            }
            ]
        }
        ],
        "max_tokens": 2000 # 800
    }

    return payload

def request_gpt4v(message, image, detail="auto"):
    payload = prepare_inputs(message, image, detail=detail)

    response_text, retry, regular_time = '', 0, 10
    while len(response_text) < 1 and retry < 1: # try multiple times if error occurs
        retry += 1
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response_text = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            time.sleep(regular_time)
            continue
    try:
        if len(response_text) < 1:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response_text = response.json()["choices"][0]["message"]["content"]
        return response_text
    except Exception as e: 
        print(e)
        raise Exception("GPT4V request error! {}".format(response.json()))
    
def collect_single_thread(target_files, img_folder, output_dir):
    # request GPT4V and save results
    print("This thread got {} images to be proccessed!".format(len(target_files)))
    for target in target_files:
        ds_name = target['dataset']
        img_file_name = target['img_id']
        target_fn = target['target_fn']

        if img_file_name in unavailable_list:
            continue

        # get image
        if ds_name == 'coco':
            this_img = coco_annos[img_file_name]
            image_path = img_folder[ds_name].format(this_img['type'], this_img['file_name'])
        elif ds_name == 'gqa':
            image_path = img_folder[ds_name].format(img_file_name + '.jpg')
        elif ds_name == 'mmvet':
            image_path = img_folder[ds_name].format(img_file_name) 
        elif ds_name == 'mmvp':
            image_path = img_folder[ds_name].format(img_file_name.split('_')[1] + '.jpg') 
        elif ds_name == 'mmmu':
            item_list = img_file_name.split('_')[1:-1]
            if len(item_list) == 1:
                sub_folder = item_list[0] 
            else:
                sub_folder = item_list[0]
                for item in item_list[1:]:
                    sub_folder += '_' + item
            image_path = img_folder[ds_name].format(sub_folder, img_file_name + '.png') 
        base64_image = encode_image_from_file(image_path)
        
        message = ""
        
        try:
            # send to GPT machines
            res = request_gpt4v(message, base64_image)     
            res = {"dataset_name": ds_name, "result": res}           
            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{target_fn}", "w") as f:
                json.dump(res, f)
        except Exception as e: 
            time.sleep(1.0)
            print(f"Error processing file '{img_file_name}': {e}")

def collect_multi_thread(output_dir, target_files, img_folder, num_tasks=1, max_retry=2):
    # query GPT4V for response
    retry = 0
    while retry < max_retry:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")
            
            # Files that have not been processed yet.
            incomplete_files = [f for f in target_files if f['target_fn'] not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # NOTE: single-thread evaluation (for debugging)
            # collect_single_thread(incomplete_files, img_folder, output_dir)

            # # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1
            
            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(part, img_folder, output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(collect_single_thread, task_args)
            retry += 1

        except Exception as e:
            retry += 1
            print("retry {}".format(retry))
            time.sleep(1.0)
            print(f"Error: {e}")

if __name__ == "__main__":    
    # input
    input_imgid_file = [
                        './coco_imageID_61k.txt',
                        './mmvp_image_ids.txt', 
                        './mmmu_image_ids.txt',
                        './mm-vet_imageID.txt', 
                        './gqa_imageID.txt',
                        ]
    dataset_name = [
                    'coco', 
                    'mmvp', 
                    'mmmu', 
                    'mmvet',
                    'gqa',
                    ]
    img_folder = {"coco": "./LLaVA_temp_20231205/playground/data/coco/{}/{}",
                  "gqa": "./GQA/raw/images/{}",
                  "mmvet": "./MM-Vet/mm-vet/images/{}",
                  "mmvp": "./MMVP/MMVP/images/{}",
                  "mmmu": "./MMMU/images/{}/{}",
                  }
    
    # output
    output_res_file = 'gpt4v_responses_auto-resolution_combined.json' # the path to save eval final combined json file
    output_dir = 'gpt4v_responses_json_folder'  # the path to save individual json files
    num_tasks = 16 # number of splits

    # eval output folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Output result json files to: {}".format(output_dir))

    # image id list
    target_files = [] # record image id
    for file_i, i_file in enumerate(input_imgid_file):
        with open(i_file, 'r') as file:
            for line in file:
                if line.strip() in unavailable_list:
                    continue
                target_files.append({'dataset': dataset_name[file_i], 'img_id': line.strip(), 'target_fn': "{}.json".format(line.strip())})
    target_files = target_files

    # start request
    start = time.time()
    collect_multi_thread(output_dir, target_files, img_folder, num_tasks=num_tasks)
    end = time.time()
    print("Used time {} for {} images with {} num_tasks!".format(end-start, len(target_files), num_tasks))
    print("Used time {} for {} images with {} num_tasks!".format(end-start, len(target_files), num_tasks))

    # merge json files
    combine_pool_files(output_dir, output_res_file)
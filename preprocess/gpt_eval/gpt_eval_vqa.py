# --------------------------------------------------------
# Evaluate VQA answers using GPT
# Adapted from:
#   https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/quantitative_evaluation/evaluate_activitynet_qa.py
#   https://github.com/yuweihao/MM-Vet/blob/main/mm-vet_evaluator.ipynb
# --------------------------------------------------------

import openai
import os
import requests
import json
import ast
from multiprocessing.pool import Pool
import pandas as pd
import random
import time
import ipdb
import argparse
from tqdm import tqdm

gpt_model = "gpt-3.5-turbo-1106" 

# Get OpenAI API Key from environment variable
api_base = "https://api.openai.com/v1"
api_key = "sk-"  # os.environ["OPENAI_API_KEY"]
# api_org = "org-"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    # "OpenAI-Organization": f"{api_org}"
}

system_prompt = '''
You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.
Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:
##INSTRUCTIONS: 
- Focus on the meaningful match between the predicted answer and the correct answer.
- Consider synonyms or paraphrases as valid matches.
- Evaluate the correctness of the prediction compared to the answer.
- <AND> in the correct answer means it is totally right only when all elements in the correct answer are present in the predicted answer.
- <OR> means it is totally right when any one element in the correct answer is present in the predicted answer.
'''

def compute_accuracy(output_json):
    combined_contents = json.load(open(output_json))
    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for key, result in combined_contents.items():
        # Computing score
        count += 1
        
        # To handle special cases like {"pred": "yes", "Score": 5} and {'pred': 'no', 'socre': 1}
        result_gpt_eval = {}
        for key, value in result[0].items():
            if isinstance(value, int) or isinstance(value, float):
                key = 'score'
                e = 0.0001
                assert value >= 0-e and value <= 5+e, f"{result[0]}. {value} is not between 0 and 5!"
            else:
                # key = key.lower()
                # assert key == 'pred', f"{key} is not 'pred'!" # AssertionError: ped is not 'pred'!
                key = "pred"
                value = value.lower()
                assert value == 'yes' or value == 'no', f"{result[0]}. {value} is not 'yes' or 'no'!"
            result_gpt_eval[key] = value
                    
        score_match = result_gpt_eval['score']
        score = int(score_match)
        score_sum += score

        # Computing accuracy
        pred = result_gpt_eval['pred']
        if "yes" in pred.lower():
            yes_count += 1
        elif "no" in pred.lower():
            no_count += 1

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)

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
    print("All evaluation completed!")

def annotate(prediction_set, target_files, output_dir):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """    
    for file in target_files:
        key = file[:-5] # strip file extension, e.g., ".json"
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']
        
         # check whether this qa has been proccessed yet
        if os.path.isfile(f"{output_dir}/{key}.json"):
            continue
        
        time.sleep(1)
         
        try:
            # Compute the correctness score
            payload = {
                "model": gpt_model,
                # "response_format": {"type": "json_object"},  # Enable JSON mode
                "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                }, 
                {
                    "role": "user",
                    "content":
                        "Please evaluate the following predicted answer given question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {answer}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                }
                ],
                "max_tokens": 800
            }
            
            # Convert response to a Python dictionary.
            response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload)
            if 'error' in response.json() and 'repetitive patterns in your prompt' in response.json()['error']['message']: # repeated tokens in model prediction
                response_message = '{"pred": "no", "score": 0}'
                print("Repeated patterns found in image {}".format(qa_set['image']))
            else:
                response_message = response.json()["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set, {"gpt_model": gpt_model}]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            time.sleep(1.0)
            print(f"response.json() = {response.json()}. Error processing file '{key}': {e}")

def eval_predictions(output_dir, target_files, prediction_set, num_tasks=1, max_retry=5):
    # retry if error occurs
    retry = 0
    while retry < max_retry:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            if 'allcombined_eval.json' in completed_files: completed_files.remove('allcombined_eval.json')
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in target_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # NOTE: single-thread evaluation (for debugging purpose, should be commented out)
            # annotate(prediction_set, incomplete_files, output_dir)

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1
            
            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)
            retry += 1

        except Exception as e:
            retry += 1
            print("retry {}".format(retry))
            time.sleep(1.0)
            print(f"Error: {e}")

def prepare_gqa(gt_path, pred_path):
    # prepare dictionary of question-answer sets for ground-truth annotations
    gt_qa = pd.read_json(path_or_buf=gt_path, lines=True) # 'question_id', 'image', 'question', 'answer'
    gt_set = {}
    for i in range(len(gt_qa)):
        question_id = str(gt_qa['question_id'][i])
        gt_set[question_id] = {'image': gt_qa['image'][i], 'q': gt_qa['question'][i], 'a': gt_qa['answer'][i]}
    
    # prepare dictionary of question-answer sets for model prediction
    pred_qa = pd.read_json(path_or_buf=pred_path, lines=True) # 'question_id', 'prompt', 'text', 'answer_id', 'model_id', 'metadata'
    pred_set = {}
    for i in range(len(pred_qa)):
        question_id = str(pred_qa['question_id'][i])
        pred_set[question_id] = {'image': gt_set[question_id]['image'], 'q': gt_set[question_id]['q'], 
                                       'a': gt_set[question_id]['a'], 'pred': pred_qa['text'][i]}
    target_files = [f"{id}.json" for id in pred_set]
    print("We will evaluate {} question-answer pairs!".format(len(target_files)))
    return pred_set, target_files

def prepare_mmmu(gt_path, pred_path):
    # prepare dictionary of question-answer sets for ground-truth annotations
    gt_qa = pd.read_json(path_or_buf=gt_path, lines=True) # 'question_id', 'image', 'question', 'answer'
    gt_set = {}
    for i in range(len(gt_qa)):
        question_id = str(gt_qa['question_id'][i])
        gt_a = gt_qa['answer'][i]
        if type(gt_a) == str:
            gt_set[question_id] = {'image': gt_qa['image'][i], 'q': gt_qa['text'][i], 'a': gt_a}
        elif type(gt_a) == list:
            gt_a_str = gt_a[0]
            for item in gt_a[1:]:
                gt_a_str += ' <OR> ' + item
            gt_set[question_id] = {'image': gt_qa['image'][i], 'q': gt_qa['text'][i], 'a': gt_a_str}
    
    # prepare dictionary of question-answer sets for model prediction
    pred_qa = pd.read_json(path_or_buf=pred_path, lines=True) # 'question_id', 'prompt', 'text', 'answer_id', 'model_id', 'metadata'
    prediction_set = {}
    for i in range(len(pred_qa)):
        question_id = str(pred_qa['question_id'][i])
        prediction_set[question_id] = {'image': gt_set[question_id]['image'], 'q': gt_set[question_id]['q'], 
                                    'a': gt_set[question_id]['a'], 'pred': pred_qa['text'][i]}
    target_files = [f"{id}.json" for id in prediction_set]
    print("We will evaluate {} question-answer pairs!".format(len(target_files)))
    return prediction_set, target_files

def main(args):
    """
    Main function to control the flow of the program.
    """
    # input arguments
    dataset = args.dataset  # 'gqa' or 'mmmu'
    if dataset == 'gqa':
        gt_path = './playground/data_VT/gpt_eval/gqa/gqa.jsonl' # the path to ground-truth annotations
    elif dataset == 'mmmu':
        gt_path = './playground/data_VT/gpt_eval/mmmu/mmmu.jsonl'  # the path to ground-truth annotations
    
    pred_path = args.pred_path
    output_dir = pred_path.rsplit('/', 1)[0] + f'/gpt_eval_{gpt_model}'
    
    output_json = '{}/allcombined_eval.json'.format(output_dir) # the path to save eval final combined json file
    num_tasks = 8 # number of splits
    
    # eval output folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Input pred jsonl: {}".format(pred_path))
    print("Ground-truth jsonl: {}".format(gt_path))
    print("Output eval json files: {}".format(output_dir))

    #############################################
    if dataset == 'gqa':
        prediction_set, target_files = prepare_gqa(gt_path, pred_path)
    elif dataset == 'mmmu':
        prediction_set, target_files = prepare_mmmu(gt_path, pred_path)

    #############################################
    eval_predictions(output_dir, target_files, prediction_set, num_tasks)

    eval_files = os.listdir(output_dir)
    if 'allcombined_eval.json' in eval_files: eval_files.remove('allcombined_eval.json')
    num_eval = len(eval_files)
    num_gt = sum(1 for line in open(gt_path))
    assert num_eval == num_gt, f"Number of eval files ({num_eval}) does not match number of ground-truth files ({num_gt})!"

    #############################################
    combine_pool_files(output_dir, output_json)

    #############################################
    compute_accuracy(output_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="", help='Dataset name: gqa or mmmu')
    parser.add_argument('--pred_path', type=str, default="", help='Path to prediction jsonl')
    args = parser.parse_args()
    print(f"args = {args}")
    main(args)
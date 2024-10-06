# --------------------------------------------------------
# Evaluate MMVP answers using GPT
# Adapted from:
#   https://github.com/tsb0601/MMVP/blob/main/scripts/gpt_grader.py
# --------------------------------------------------------
import json
import openai
import re
import time
import requests
import pandas as pd
import ipdb
import argparse
import os
from tqdm import tqdm

# openai
api_base = "https://api.openai.com/v1"
api_key = "sk-"  # os.environ["OPENAI_API_KEY"]
# api_org = "org-"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    # "OpenAI-Organization": f"{api_org}"
}

gpt_model = "gpt-3.5-turbo-1106" 

NUM_SECONDS_TO_SLEEP = 10
# Define a function to query the OpenAI API and evaluate the answer
def get_yes_no_answer(question):
    while True:
        try:
            payload = {
                "model": gpt_model,
                "messages": [
                {
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer. Please answer in only yes or no.'
                }, 
                {
                    'role': 'user',
                    'content': question,
                }
                ],
                "temperature": 0.2,
            }
            response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload)
            response = response.json()
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(f"response.json() = {response.json()}. " + e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    answer = response['choices'][0]['message']['content']
    yes_no_regex = re.compile(r"^(yes|no)$", re.IGNORECASE)

    if yes_no_regex.match(answer):
        return answer.lower()
    else:
        return "Could not determine yes or no."



def main(args):
    # input arguments
    answer_file = args.pred_path
    output_dir = answer_file.rsplit('/', 1)[0] + f'/gpt_eval_{gpt_model}'
    gt_file = "./playground/data_VT/gpt_eval/mmvp/mmvp.jsonl"

    # eval output folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_correct, num_total = 0, 0
    # Continue with the processing of the JSONL file
    gt_qa = pd.read_json(path_or_buf=gt_file, lines=True)
    pred_qa = pd.read_json(path_or_buf=answer_file, lines=True)
    for line_i in tqdm(range(len(pred_qa))):
        # prepare inputs
        assert pred_qa["question_id"][line_i] == gt_qa["question_id"][line_i]
        question_id = int(pred_qa["question_id"][line_i])
        
        # check whether this qa has been proccessed yet
        if os.path.isfile(f"{output_dir}/{question_id}.json"):
            continue
        
        question, correct_answer, model_response = gt_qa["text"][line_i], gt_qa["answers"][line_i], pred_qa["text"][line_i]
        question4gpt = f"Given the following question {question}, the correct answer is {correct_answer}. Does the following answer correctly answers the question, answer:{model_response}?"
        
        # evaluate model predictions
        retry = 0
        while retry <= 3:
            retry += 1
            try:
                gpt_grade = get_yes_no_answer(question4gpt)
            except Exception as e:
                print("Retry {} for question id {}".format(retry, gt_qa["question_id"][line_i]))
                time.sleep(NUM_SECONDS_TO_SLEEP)
                continue
        result_qa_pair = {'question_id': question_id, 'question': question, 'model_response': model_response, 'gpt_grade': str(gpt_grade), 'gpt_model': gpt_model}
        # result_qa_pair = {'question_id': 1, 'question': "Are the butterfly's wings closer to being open or closed?", 'model_response': "The butterfly's wings are open, showcasing the patterns and symmetry.", 'gpt_grade': 'yes'}
        
        # Save the question-answer pairs to a json file.
        with open(f"{output_dir}/{question_id}.json", "w") as f:
            json.dump(result_qa_pair, f, indent=4)
                
    # accumulate results and compute accuracy
    index, round_correct = 0, 0
    completed_files = sorted(os.listdir(output_dir))
    for fn in completed_files:
        # print(f"fn = {fn}")
        res_dict = json.load(open(f"{output_dir}/{fn}",'r'))
        gpt_grade = res_dict['gpt_grade']

        index += 1

        if gpt_grade=="yes":
            round_correct += 1
        if index == 2:
            index = 0
            if round_correct == 2:
                num_correct += 1
            round_correct = 0

            num_total += 1
    
    print("Input pred jsonl: {}".format(answer_file))
    print("Ground-truth jsonl: {}".format(gt_file))
    print("Output eval json files: {}".format(output_dir))

    print(f"The accuracy is {num_correct/num_total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default="", help='Path to prediction jsonl')
    args = parser.parse_args()
    print(f"args = {args}")
    main(args)
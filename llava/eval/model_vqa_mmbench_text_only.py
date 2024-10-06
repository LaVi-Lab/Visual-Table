import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

import math


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    if args.with_gen_vt:
        visual_table_prompt_pre = "Visual table:"
        visual_table_prompt_post = "Based on the given image and given visual table, answer the following question:"
        dict_image_vt = {}
        with open(args.gen_vt_path, 'r') as file:
            for line in file:
                json_data = json.loads(line)
                imageID = json_data['question_id']
                text = json_data['text']
                dict_image_vt[imageID] = text
    elif args.with_gen_detailed_cap:
        detailed_cap_prompt_pre = "Detailed caption:"
        detailed_cap_prompt_post = "Based on the given image and given detailed caption, answer the following question:"
        dict_image_detailed_cap = {}
        with open(args.gen_detailed_cap_path, 'r') as file:
            for line in file:
                json_data = json.loads(line)
                imageID = json_data['question_id']
                text = json_data['text']
                dict_image_detailed_cap[imageID] = text
    elif args.with_gen_sg:
        sg_prompt_pre = "Scene graph:"
        sg_prompt_post = "Based on the given image and given scene graph, answer the following question:"
        dict_image_sg = {}
        with open(args.gen_sg_path, 'r') as file:
            for line in file:
                json_data = json.loads(line)
                imageID = json_data['question_id']
                text = json_data['text']
                dict_image_sg[imageID] = text
    elif args.with_gen_blip2cap:
        blip2cap_prompt_pre = "Caption:"
        blip2cap_prompt_post = "Based on the given image and given caption, answer the following question:"
        dict_image_blip2cap = {}
        with open(args.gen_blip2cap_path, 'r') as file:
            for line in file:
                json_data = json.loads(line)
                imageID = json_data['question_id']
                text = json_data['text']
                dict_image_blip2cap[imageID] = text

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            
            if args.with_gen_vt:
                assert f'{idx}.jpg' in dict_image_vt
                question = visual_table_prompt_pre + '\n' + \
                                dict_image_vt[f'{idx}.jpg'] + '\n' + \
                                visual_table_prompt_post + '\n' + \
                                question
            elif args.with_gen_detailed_cap:
                assert f'{idx}.jpg' in dict_image_detailed_cap
                question = detailed_cap_prompt_pre + '\n' + \
                                dict_image_detailed_cap[f'{idx}.jpg'] + '\n' + \
                                detailed_cap_prompt_post + '\n' + \
                                question   
            elif args.with_gen_sg:
                assert f'{idx}.jpg' in dict_image_sg
                question = sg_prompt_pre + '\n' + \
                                dict_image_sg[f'{idx}.jpg'] + '\n' + \
                                sg_prompt_post + '\n' + \
                                question   
            elif args.with_gen_blip2cap:
                assert f'{idx}.jpg' in dict_image_blip2cap
                question = blip2cap_prompt_pre + '\n' + \
                                dict_image_blip2cap[f'{idx}.jpg'] + '\n' + \
                                blip2cap_prompt_post + '\n' + \
                                question   

            qs = cur_prompt = question

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    
    parser.add_argument("--with_gen_vt", action="store_true")
    parser.add_argument("--gen_vt_path", type=str, default="playground/data/eval/eval_images_gen_vt/mmbench_gen_vt/llava13b_mix199k_bs128_gen_vt_61k_epoch3_bs128/merge.jsonl")
    
    parser.add_argument("--with_gen_detailed_cap", action="store_true")
    parser.add_argument("--gen_detailed_cap_path", type=str, default="")
    
    parser.add_argument("--with_gen_sg", action="store_true")
    parser.add_argument("--gen_sg_path", type=str, default="")
    
    parser.add_argument("--with_gen_blip2cap", action="store_true")
    parser.add_argument("--gen_blip2cap_path", type=str, default="")

    args = parser.parse_args()

    eval_model(args)

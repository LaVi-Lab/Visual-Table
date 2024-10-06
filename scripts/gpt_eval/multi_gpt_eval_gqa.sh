#!/bin/bash

dataset="gqa"
pred_paths=(
# "playground/data/eval/gqa/answers/gqa_with_vt_cap/llava13b_mix177k_with_vt_cap/gqa_with_vt_cap_llava13b_mix177k_with_vt_cap.jsonl"
)

mkdir -p scripts/log/gpt_eval

for pred_path in "${pred_paths[@]}"; do
    log_file="scripts/log/gpt_eval/$(basename "$pred_path" .jsonl).txt"
    command="python preprocess/gpt_eval/gpt_eval_vqa.py --dataset $dataset --pred_path $pred_path 2>&1 | tee -a $log_file"
    echo "$command"
    eval "$command"
    wait
done
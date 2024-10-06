#!/bin/bash

dataset="mmmu"
pred_paths=(
"./playground/data_VT/eval/mmmu/answers/mmmu_with_VTGenerator-13B_gen_vt/LLaVA-VT-13B/mmmu_with_VTGenerator-13B_gen_vt_LLaVA-VT-13B.jsonl"
)

mkdir -p scripts/log/gpt_eval

for pred_path in "${pred_paths[@]}"; do
    log_file="scripts/log/gpt_eval/$(basename "$pred_path" .jsonl).txt"
    command="python preprocess/gpt_eval/gpt_eval_vqa.py --dataset $dataset --pred_path $pred_path 2>&1 | tee -a $log_file"
    echo "$command"
    eval "$command"
    wait
done

# Input pred jsonl: ./playground/data_VT/eval/mmmu/answers/mmmu_with_VTGenerator-13B_gen_vt/LLaVA-VT-13B/mmmu_with_VTGenerator-13B_gen_vt_LLaVA-VT-13B.jsonl
# Ground-truth jsonl: ./playground/data_VT/gpt_eval/mmmu/mmmu.jsonl
# Output eval json files: ./playground/data_VT/eval/mmmu/answers/mmmu_with_VTGenerator-13B_gen_vt/LLaVA-VT-13B/gpt_eval_gpt-3.5-turbo-1106
# We will evaluate 855 question-answer pairs!
# completed_files: 855
# incomplete_files: 0
# All evaluation completed!
# Yes count: 358
# No count: 497
# Accuracy: 0.41871345029239765
# Average score: 2.8526315789473684
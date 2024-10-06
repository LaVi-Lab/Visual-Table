#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/LLaVA-VT/eval/vizwiz/vizwiz.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/vizwiz_with_${VTGenerator}_gen_vt_${Model}.txt

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# VTGenerator="VTGenerator-13B"
# Model="LLaVA-VT-13B"
if [ $# -ne 2 ]; then
  echo "Usage: $0 <VTGenerator> <Model> "
  exit 1
fi

VTGenerator=$1
Model=$2

VT_PATH="./playground/data_VT/eval_images_gen_vt/vizwiz_gen_vt/${VTGenerator}/merge.jsonl"
SPLIT=vizwiz_with_${VTGenerator}_gen_vt

# 1. merge_eval_dataset_with_VT
python ./preprocess/merge_with_VT/merge_eval_dataset_with_VT.py \
    --dataset_path "./playground/data/eval/vizwiz/llava_test.jsonl" \
    --dataset_vt_path ${VT_PATH} \
    --dataset_with_vt_path "./playground/data_VT/eval/vizwiz/${SPLIT}.jsonl"

# 2. infer LLaVA-VT-13B on eval_dataset_with_VT
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints/$Model \
        --question-file ./playground/data_VT/eval/vizwiz/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/vizwiz/test \
        --answers-file ./playground/data_VT/eval/vizwiz/answers/$SPLIT/$Model/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait

output_file=./playground/data_VT/eval/vizwiz/answers/$SPLIT/$Model/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data_VT/eval/vizwiz/answers/$SPLIT/$Model/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

output_file2=./playground/data_VT/eval/vizwiz/answers_upload/$SPLIT/$Model/${Model}.json
mkdir -p ./playground/data_VT/eval/vizwiz/answers_upload/$SPLIT/$Model

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file $output_file \
    --result-upload-file ${output_file2}

# 3. eval on official eval server
# https://eval.ai/web/challenges/challenge-page/2185/submission
# Phase: test-standard2024-VQA
# ./playground/data_VT/eval/vizwiz/answers_upload/vizwiz_with_VTGenerator-13B_gen_vt/LLaVA-VT-13B/LLaVA-VT-13B.json
# [{"test": {"overall": 57.38, "other": 48.82, "unanswerable": 75.73, "yes/no": 79.6, "number": 48.46}}]
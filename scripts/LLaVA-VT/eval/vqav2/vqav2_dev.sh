#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/LLaVA-VT/eval/vqav2/vqav2_dev.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/vqav2_dev_with_${VTGenerator}_gen_vt_${Model}.txt

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

VT_PATH="./playground/data_VT/eval_images_gen_vt/vqav2_gen_vt/${VTGenerator}/merge.jsonl"
SPLIT=vqav2_dev_with_${VTGenerator}_gen_vt

# 1. merge_eval_dataset_with_VT
python ./preprocess/merge_with_VT/merge_eval_dataset_with_VT.py \
    --dataset_path "./playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl" \
    --dataset_vt_path ${VT_PATH} \
    --dataset_with_vt_path "./playground/data_VT/eval/vqav2/${SPLIT}.jsonl"

# 2. infer LLaVA-VT-13B on eval_dataset_with_VT
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints/$Model \
        --question-file ./playground/data_VT/eval/vqav2/${SPLIT}.jsonl \
        --image-folder ./playground/data/coco/test2015 \
        --answers-file ./playground/data_VT/eval/vqav2/answers/$SPLIT/$Model/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait

output_file=./playground/data_VT/eval/vqav2/answers/$SPLIT/$Model/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data_VT/eval/vqav2/answers/$SPLIT/$Model/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --Model $Model

# 3. eval on official eval server
# https://eval.ai/web/challenges/challenge-page/830/submission
# Phase: Test-Dev Phase
# ./playground/data_VT/eval/vqav2/answers_upload/vqav2_dev_with_VTGenerator-13B_gen_vt/LLaVA-VT-13B.json
# [{"test-dev": {"yes/no": 93.9, "number": 63.41, "other": 73.36, "overall": 80.69}}]

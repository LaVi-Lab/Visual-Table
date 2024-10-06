#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/LLaVA-VT/eval/gqa/gqa.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/gqa_full_with_${VTGenerator}_gen_vt_${Model}.txt

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

VT_PATH="./playground/data_VT/eval_images_gen_vt/gqa_gen_vt/${VTGenerator}/merge.jsonl"
SPLIT=gqa_with_${VTGenerator}_gen_vt

# 1. merge_eval_dataset_with_VT
python ./preprocess/merge_with_VT/merge_eval_dataset_with_VT.py \
    --dataset_path "./playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl" \
    --dataset_vt_path ${VT_PATH} \
    --dataset_with_vt_path "./playground/data_VT/eval/gqa/${SPLIT}.jsonl"

# 2. infer LLaVA-VT-13B on eval_dataset_with_VT
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints/$Model \
        --question-file ./playground/data_VT/eval/gqa/${SPLIT}.jsonl \
        --image-folder /path/to/GQA/images \
        --answers-file ./playground/data_VT/eval/gqa/answers/$SPLIT/$Model/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait


output_file=./playground/data_VT/eval/gqa/answers/$SPLIT/$Model/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data_VT/eval/gqa/answers/$SPLIT/$Model/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# 3. eval gqa
GQADIR="./playground/data_VT/eval/gqa/data"
# download from: https://cs.stanford.edu/people/dorarad/gqa/evaluate.html
# fix some bugs of the official codes following: 
# https://github.com/haotian-liu/LLaVA/issues/584
# https://github.com/haotian-liu/LLaVA/issues/625

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cp $GQADIR/testdev_balanced_predictions.json ./playground/data_VT/eval/gqa/answers/$SPLIT/$Model

cd $GQADIR
python eval/eval.py --tier testdev_balanced --questions testdev_balanced_questions.json

# ./playground/data_VT/eval/gqa/answers/gqa_with_VTGenerator-13B_gen_vt/LLaVA-VT-13B/testdev_balanced_predictions.json
# Binary: 81.27%
# Open: 49.30%
# Accuracy: 63.98%
# Validity: 0.00%
# Plausibility: 0.00%
# Distribution: 1.68 (lower is better)

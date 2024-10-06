#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/LLaVA-VT/eval/textvqa/textvqa.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/textvqa_with_${VTGenerator}_gen_vt_${Model}.txt

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

VT_PATH="./playground/data_VT/eval_images_gen_vt/textvqa_gen_vt/${VTGenerator}/merge.jsonl"
SPLIT=textvqa_with_${VTGenerator}_gen_vt

# 1. merge_eval_dataset_with_VT
python ./preprocess/merge_with_VT/merge_eval_dataset_with_VT.py \
    --dataset_path "./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
    --dataset_vt_path ${VT_PATH} \
    --dataset_with_vt_path "./playground/data_VT/eval/textvqa/${SPLIT}.jsonl"

# 2. infer LLaVA-VT-13B on eval_dataset_with_VT
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints/$Model \
        --question-file ./playground/data_VT/eval/textvqa/${SPLIT}.jsonl \
        --image-folder ./playground/data/eval/textvqa/train_images \
        --answers-file ./playground/data_VT/eval/textvqa/answers/$SPLIT/$Model/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait

output_file=./playground/data_VT/eval/textvqa/answers/$SPLIT/$Model/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data_VT/eval/textvqa/answers/$SPLIT/$Model/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# 3. eval textvqa
python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file

# Samples: 5000
# Accuracy: 61.16%
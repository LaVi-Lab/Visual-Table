#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/VTGenerator/infer/train_images_gen_vt/llava_instruct_mix665k_vg_gen_vt.sh VTGenerator-13B 

if [ $# -ne 1 ]; then
  echo "Usage: $0 <CKPT>"
  exit 1
fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
SPLIT="llava_instruct_mix665k_vg_gen_vt"
OUT="./playground/data/eval/train_images_gen_vt"

mkdir -p ${OUT}/${SPLIT}/${CKPT}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
    --model-path ./checkpoints/$CKPT \
    --question-file ./playground/data/eval/train_images_gen_vt/$SPLIT.jsonl \
    --image-folder /path/to/vg_images \
    --answers-file ${OUT}/${SPLIT}/${CKPT}/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &
done

wait

output_file=${OUT}/${SPLIT}/${CKPT}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${OUT}/${SPLIT}/${CKPT}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/LLaVA-VT/eval/pope/pope.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/pope_with_${VTGenerator}_gen_vt_${Model}.txt

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

VT_PATH="./playground/data_VT/eval_images_gen_vt/pope_gen_vt/${VTGenerator}/merge.jsonl"
SPLIT=pope_with_${VTGenerator}_gen_vt

# 1. merge_eval_dataset_with_VT
python ./preprocess/merge_with_VT/merge_eval_dataset_with_VT.py \
    --dataset_path "./playground/data/eval/pope/llava_pope_test.jsonl" \
    --dataset_vt_path ${VT_PATH} \
    --dataset_with_vt_path "./playground/data_VT/eval/pope/${SPLIT}.jsonl"

# 2. infer LLaVA-VT-13B on eval_dataset_with_VT
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints/$Model \
        --question-file ./playground/data_VT/eval/pope/$SPLIT.jsonl \
        --image-folder ./playground/data/coco/val2014 \
        --answers-file ./playground/data_VT/eval/pope/answers/$SPLIT/$Model/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait

output_file=./playground/data_VT/eval/pope/answers/$SPLIT/$Model/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data_VT/eval/pope/answers/$SPLIT/$Model/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# 3. eval pope
python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data_VT/eval/pope/pope_coco_commitID_e3e39262c85a6a83f26cf5094022a782cb0df58d \
    --question-file ./playground/data_VT/eval/pope/$SPLIT.jsonl \
    --result-file $output_file

# ./playground/data_VT/eval/pope/answers/pope_with_VTGenerator-13B_gen_vt/LLaVA-VT-13B/merge.jsonl
# Category: adversarial, # samples: 3000
# TP      FP      TN      FN
# 1210    121     1379    290
# Accuracy: 0.863
# Precision: 0.9090909090909091
# Recall: 0.8066666666666666
# F1 score: 0.8548216178028964
# Yes ratio: 0.44366666666666665
# 0.855, 0.863, 0.909, 0.807, 0.444
# ====================================
# Category: random, # samples: 2910
# TP      FP      TN      FN
# 1210    25      1385    290
# Accuracy: 0.8917525773195877
# Precision: 0.979757085020243
# Recall: 0.8066666666666666
# F1 score: 0.8848263254113345
# Yes ratio: 0.42439862542955326
# 0.885, 0.892, 0.980, 0.807, 0.424
# ====================================
# Category: popular, # samples: 3000
# TP      FP      TN      FN
# 1210    62      1438    290
# Accuracy: 0.8826666666666667
# Precision: 0.9512578616352201
# Recall: 0.8066666666666666
# F1 score: 0.8730158730158729
# Yes ratio: 0.424
# 0.873, 0.883, 0.951, 0.807, 0.424
# ====================================
# avg_f1 =  0.870887938743368
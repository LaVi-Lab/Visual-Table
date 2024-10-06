#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0 bash scripts/LLaVA-VT/eval/mmmu/mmmu.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/mmmu_with_${VTGenerator}_gen_vt_${Model}.txt

# VTGenerator="VTGenerator-13B"
# Model="LLaVA-VT-13B"
if [ $# -ne 2 ]; then
  echo "Usage: $0 <VTGenerator> <Model> "
  exit 1
fi

VTGenerator=$1
Model=$2

VT_PATH="./playground/data_VT/eval_images_gen_vt/mmmu_gen_vt/${VTGenerator}/merge.jsonl"
SPLIT=mmmu_with_${VTGenerator}_gen_vt

# 1. merge_eval_dataset_with_VT
python ./preprocess/merge_with_VT/merge_eval_dataset_with_VT.py \
    --dataset_path "./playground/data_VT/eval/mmmu/mmmu.jsonl" \
    --dataset_vt_path ${VT_PATH} \
    --dataset_with_vt_path "./playground/data_VT/eval/mmmu/${SPLIT}.jsonl"

# 2. infer LLaVA-VT-13B on eval_dataset_with_VT
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/$Model \
    --question-file ./playground/data_VT/eval/mmmu/$SPLIT.jsonl \
    --image-folder /path/to/MMMU/images \
    --answers-file ./playground/data_VT/eval/mmmu/answers/$SPLIT/$Model/${SPLIT}_${Model}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &
wait

# 3. eval in ./scripts/gpt_eval/multi_gpt_eval_mmmu.sh
# pred_paths=(
#   "./playground/data_VT/eval/mmmu/answers/$SPLIT/$Model/${SPLIT}_${Model}.jsonl"
# )
# Yes count: 358
# No count: 497
# Accuracy: 0.41871345029239765
# Average score: 2.8526315789473684

#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0 bash scripts/LLaVA-VT/eval/mmvp_mc/mmvp_mc.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/mmvp_mc_with_${VTGenerator}_gen_vt_${Model}.txt

# VTGenerator="VTGenerator-13B"
# Model="LLaVA-VT-13B"
if [ $# -ne 2 ]; then
  echo "Usage: $0 <VTGenerator> <Model> "
  exit 1
fi

VTGenerator=$1
Model=$2

VT_PATH="./playground/data_VT/eval_images_gen_vt/mmvp_gen_vt/${VTGenerator}/merge.jsonl"
SPLIT=mmvp_mc_with_${VTGenerator}_gen_vt

# 1. merge_eval_dataset_with_VT
python ./preprocess/merge_with_VT/merge_eval_dataset_with_VT.py \
    --dataset_path "./playground/data_VT/eval/mmvp_mc/mmvp_mc.jsonl" \
    --dataset_vt_path ${VT_PATH} \
    --dataset_with_vt_path "./playground/data_VT/eval/mmvp_mc/${SPLIT}.jsonl"

# 2. infer LLaVA-VT-13B on eval_dataset_with_VT
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/$Model \
    --question-file ./playground/data_VT/eval/mmvp_mc/$SPLIT.jsonl \
    --image-folder /path/to/MMVP/images \
    --answers-file ./playground/data_VT/eval/mmvp_mc/answers/$SPLIT/$Model/${SPLIT}_${Model}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 &
wait

# 3. eval mmvp_mc
python ./scripts/LLaVA-VT/eval/mmvp_mc/eval_mmvp_mc_acc.py \
  --path_pred "./playground/data_VT/eval/mmvp_mc/answers/$SPLIT/$Model/${SPLIT}_${Model}.jsonl" \
  --path_gt "./playground/data_VT/eval/mmvp_mc/mmvp_mc.jsonl"

# path_pred: ./playground/data_VT/eval/mmvp_mc/answers/mmvp_mc_with_VTGenerator-13B_gen_vt/LLaVA-VT-13B/mmvp_mc_with_VTGenerator-13B_gen_vt_LLaVA-VT-13B.jsonl
# path_gt: ./playground/data_VT/eval/mmvp_mc/mmvp_mc.jsonl
# Accuracy: 0.36666666666666664
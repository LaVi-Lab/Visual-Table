#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0 bash scripts/LLaVA-VT/eval/mmvet/mmvet.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/mmvet_with_${VTGenerator}_gen_vt_${Model}.txt

# VTGenerator="VTGenerator-13B"
# Model="LLaVA-VT-13B"
if [ $# -ne 2 ]; then
  echo "Usage: $0 <VTGenerator> <Model> "
  exit 1
fi

VTGenerator=$1
Model=$2

VT_PATH="./playground/data_VT/eval_images_gen_vt/mmvet_gen_vt/${VTGenerator}/merge.jsonl"
SPLIT=mmvet_with_${VTGenerator}_gen_vt

# 1. merge_eval_dataset_with_VT
# def dataset_with_vt(dataset_path, dataset_vt_path, dataset_with_vt_path)
python ./preprocess/merge_with_VT/merge_eval_dataset_with_VT.py \
    --dataset_path "./playground/data/eval/mm-vet/llava-mm-vet.jsonl" \
    --dataset_vt_path ${VT_PATH} \
    --dataset_with_vt_path "./playground/data_VT/eval/mm-vet/${SPLIT}.jsonl"

# 2. infer LLaVA-VT-13B on eval_dataset_with_VT
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/$Model \
    --question-file ./playground/data_VT/eval/mm-vet/${SPLIT}.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data_VT/eval/mm-vet/$SPLIT/answers/${Model}/${SPLIT}_${Model}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
wait

mkdir -p ./playground/data_VT/eval/mm-vet/$SPLIT/results/${Model}

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data_VT/eval/mm-vet/$SPLIT/answers/${Model}/${SPLIT}_${Model}.jsonl \
    --dst ./playground/data_VT/eval/mm-vet/$SPLIT/results/${Model}/${SPLIT}_${Model}.json

# 3. eval on official eval server
# https://huggingface.co/spaces/whyu/MM-Vet_Evaluator
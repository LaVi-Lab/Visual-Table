#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0 bash scripts/LLaVA-VT/eval/scienceqa/scienceqa.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/scienceqa_with_${VTGenerator}_gen_vt_${Model}.txt

# VTGenerator="VTGenerator-13B"
# Model="LLaVA-VT-13B"
if [ $# -ne 2 ]; then
  echo "Usage: $0 <VTGenerator> <Model> "
  exit 1
fi

VTGenerator=$1
Model=$2

VT_PATH="./playground/data_VT/eval_images_gen_vt/scienceqa_gen_vt/${VTGenerator}/merge.jsonl"
SPLIT=scienceqa_with_${VTGenerator}_gen_vt

# 1. merge_eval_dataset_with_VT
python ./preprocess/merge_with_VT/merge_scienceqa_with_VT.py \
    --dataset_path "./playground/data/eval/scienceqa/llava_test_CQM-A.json" \
    --dataset_vt_path ${VT_PATH} \
    --dataset_with_vt_path "./playground/data_VT/eval/scienceqa/${SPLIT}.json"

# 2. infer LLaVA-VT-13B on eval_dataset_with_VT
python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/$Model \
    --question-file ./playground/data_VT/eval/scienceqa/${SPLIT}.json \
    --image-folder ./playground/data/eval/scienceqa/test \
    --answers-file ./playground/data_VT/eval/scienceqa/answers/$SPLIT/$Model/$Model.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

# 3. eval scienceqa-img
# llava/eval/eval_science_qa.py 
#     base_dir = args.base_dir
#     split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
#     problems = json.load(open(os.path.join(base_dir, "problems.json")))

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data_VT/eval/scienceqa/answers/$SPLIT/$Model/$Model.jsonl \
    --output-file ./playground/data_VT/eval/scienceqa/answers/$SPLIT/$Model/${Model}_output.jsonl \
    --output-result ./playground/data_VT/eval/scienceqa/answers/$SPLIT/$Model/${Model}_result.json

# Total: 4241, Correct: 3189, Accuracy: 75.19%, IMG-Accuracy: 72.58%
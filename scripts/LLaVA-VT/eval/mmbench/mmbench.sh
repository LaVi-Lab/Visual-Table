#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0 bash scripts/LLaVA-VT/eval/mmbench/mmbench.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/mmbench_with_${VTGenerator}_gen_vt_${Model}.txt

# VTGenerator="VTGenerator-13B"
# Model="LLaVA-VT-13B"
if [ $# -ne 2 ]; then
  echo "Usage: $0 <VTGenerator> <Model> "
  exit 1
fi

VTGenerator=$1
Model=$2
VT_PATH="./playground/data_VT/eval_images_gen_vt/mmbench_gen_vt/${VTGenerator}/merge.jsonl"
SPLIT=mmbench_with_${VTGenerator}_gen_vt

mkdir -p playground/data_VT/eval/mmbench/answers/$SPLIT/$Model
mkdir -p playground/data_VT/eval/mmbench/answers_upload/$SPLIT/$Model

# 1. merge_eval_dataset_with_VT
# 2. infer LLaVA-VT-13B on eval_dataset_with_VT
python -m llava.eval.model_vqa_mmbench \
    --model-path ./checkpoints/$Model \
    --question-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
    --answers-file ./playground/data_VT/eval/mmbench/answers/$SPLIT/$Model/$Model.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --with_gen_vt \
    --gen_vt_path ${VT_PATH}
wait

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
    --result-dir ./playground/data_VT/eval/mmbench/answers/$SPLIT/$Model \
    --upload-dir ./playground/data_VT/eval/mmbench/answers_upload/$SPLIT/$Model \
    --experiment $Model

# 3. eval on official eval server
# https://mmbench.opencompass.org.cn/mmbench-submission
# ./playground/data_VT/eval/mmbench/answers_upload/mmbench_with_VTGenerator-13B_gen_vt/LLaVA-VT-13B/LLaVA-VT-13B.xlsx
# key	value
# A_Overall (dev)	0.6941580756013745
# B_AR (dev)	0.7185929648241206
# B_CP (dev)	0.7972972972972973
# B_FP-C (dev)	0.6293706293706294
# B_FP-S (dev)	0.7201365187713311
# B_LR (dev)	0.423728813559322
# B_RR (dev)	0.6782608695652174
# C_action_recognition (dev)	0.9074074074074074
# C_attribute_comparison (dev)	0.6136363636363636
# C_attribute_recognition (dev)	0.8378378378378378
# C_celebrity_recognition (dev)	0.8181818181818182
# C_function_reasoning (dev)	0.8227848101265823
# C_future_prediction (dev)	0.575
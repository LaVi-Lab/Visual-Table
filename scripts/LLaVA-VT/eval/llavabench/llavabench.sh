#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0 bash scripts/LLaVA-VT/eval/llavabench/llavabench.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/llavabench_with_${VTGenerator}_gen_vt_${Model}.txt

# SPLIT=$1
# Model=$2

# VTGenerator="VTGenerator-13B"
# Model="LLaVA-VT-13B"
if [ $# -ne 2 ]; then
  echo "Usage: $0 <VTGenerator> <Model> "
  exit 1
fi

VTGenerator=$1
Model=$2

VT_PATH="./playground/data_VT/eval_images_gen_vt/llavabench_gen_vt/${VTGenerator}/merge.jsonl"
SPLIT=llavabench_with_${VTGenerator}_gen_vt

# 1. merge_eval_dataset_with_VT
# def dataset_with_vt(dataset_path, dataset_vt_path, dataset_with_vt_path)
python ./preprocess/merge_with_VT/merge_eval_dataset_with_VT.py \
    --dataset_path "./playground/data/eval/llava-bench-in-the-wild/questions.jsonl" \
    --dataset_vt_path ${VT_PATH} \
    --dataset_with_vt_path "./playground/data_VT/eval/llava-bench-in-the-wild/${SPLIT}.jsonl"

# 2. infer LLaVA-VT-13B on eval_dataset_with_VT
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/$Model \
    --question-file ./playground/data_VT/eval/llava-bench-in-the-wild/${SPLIT}.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data_VT/eval/llava-bench-in-the-wild/answers/$SPLIT/$Model/$Model.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
wait

# The evaluation code provided by llavabench requires openai==0.28.0
# Please change to conda env with openai==0.28.0
# And then excuse the following command
version=$(pip list | grep openai | awk '{print $2}')
if [ $version != "0.28.0" ]; then
    echo "Please change to conda env with openai==0.28.0; And then excuse the following command."
    exit 1
fi

gpt_model="gpt-3.5-turbo-1106"

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews/$SPLIT/$Model
python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$SPLIT/$Model/$Model.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$SPLIT/$Model/${Model}_${gpt_model}.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$SPLIT/$Model/${Model}_${gpt_model}.jsonl

# python llava/eval/summarize_gpt_review.py -f ./playground/data_VT/eval/llava-bench-in-the-wild/reviews/llavabench_with_VTGenerator-13B_gen_vt/LLaVA-VT-13B/LLaVA-VT-13B_gpt-3.5-turbo-1106.jsonl
# LLaVA-VT-13B_gpt-3.5-turbo-1106
# all 89.1 76.2 67.8
# llava_bench_complex 95.2 75.0 71.4
# llava_bench_conv 92.1 81.8 75.3
# llava_bench_detail 73.2 72.0 52.7
# =================================
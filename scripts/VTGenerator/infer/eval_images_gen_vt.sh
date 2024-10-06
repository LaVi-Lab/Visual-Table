#!/bin/bash

# usage: 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/VTGenerator/infer/eval_images_gen_vt.sh VTGenerator-13B

if [ $# -ne 1 ]; then
  echo "Usage: $0 <CKPT>"
  exit 1
fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1

# download eval datasets and their images first
# following https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md

# for mmmu and mmvp, download dataset from their official repos

# for mmbench, decode_base64_to_image
# store images in ./playground/data/eval/mmbench/images/mmbench_dev_20230712
python ./preprocess/mmbench/convert_mmbench_images.py \
  --data_path "./playground/data/eval/mmbench/mmbench_dev_20230712.tsv" \
  --image_path "./playground/data/eval/mmbench/images/mmbench_dev_20230712"

SPLITS=(
"mmvet_gen_vt"
"gqa_gen_vt"
"mmmu_gen_vt"
"mmvp_gen_vt"
"llavabench_gen_vt"
"mmbench_gen_vt"
"pope_gen_vt"
"scienceqa_gen_vt"
"textvqa_gen_vt"
"vizwiz_gen_vt"
"vqav2_gen_vt"
)

# please modify image path
IMAGE_FOLDERS=(
"/path/to/mm-vet/images"
"/path/to/GQA/raw/images"
"/path/to/MMMU/images"
"/path/to/MMVP/MMVP/images"
"/path/to/llava-bench-in-the-wild/images"
"./playground/data/eval/mmbench/images/mmbench_dev_20230712"
"/path/to/coco/val2014"
"/path/to/scienceqa/test"
"/path/to/textvqa/train_images"
"/path/to/vizwiz/test"
"/path/to/vqav2/test2015"
)


OUT="./playground/data_VT/eval_images_gen_vt"

for ((i=0; i<${#SPLITS[@]}; i++)); do
  SPLIT=${SPLITS[$i]}
  IMAGE_FOLDER=${IMAGE_FOLDERS[$i]}

  echo $SPLIT
  echo ${IMAGE_FOLDER}

  mkdir -p ${OUT}/${SPLIT}/${CKPT}

  for IDX in $(seq 0 $((CHUNKS-1))); do
      CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
      --model-path ./checkpoints/$CKPT \
      --question-file $OUT/$SPLIT.jsonl \
      --image-folder ${IMAGE_FOLDER} \
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
done
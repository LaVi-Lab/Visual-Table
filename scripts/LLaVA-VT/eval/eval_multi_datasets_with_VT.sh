VTGenerator="VTGenerator-13B"
Model="LLaVA-VT-13B"

mkdir -p scripts/log/eval_multi_datasets_with_VT

# before this script
# 1. download evaluation images and eval.zip (following https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)
# 2. utilize ./scripts/VTGenerator/infer/eval_images_gen_vt.sh to generate VT for each dataset
# or directly utilize the provided VT for each dataset from ./playground/data_VT/eval_images_gen_vt

# for each dataset
# 1. merge_eval_dataset_with_VT
# 2. infer LLaVA-VT-13B on eval_dataset_with_VT
# 3. [optional] GPT-assisted evaluation: mmvet, llavabench, mmmu

CUDA_VISIBLE_DEVICES=0 bash scripts/LLaVA-VT/eval/mmvet/mmvet.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/mmvet_with_${VTGenerator}_gen_vt_${Model}.txt

CUDA_VISIBLE_DEVICES=0 bash scripts/LLaVA-VT/eval/llavabench/llavabench.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/llavabench_with_${VTGenerator}_gen_vt_${Model}.txt

CUDA_VISIBLE_DEVICES=0 bash scripts/LLaVA-VT/eval/mmmu/mmmu.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/mmmu_with_${VTGenerator}_gen_vt_${Model}.txt

CUDA_VISIBLE_DEVICES=0 bash scripts/LLaVA-VT/eval/mmbench/mmbench.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/mmbench_with_${VTGenerator}_gen_vt_${Model}.txt

CUDA_VISIBLE_DEVICES=0 bash scripts/LLaVA-VT/eval/mmvp_mc/mmvp_mc.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/mmvp_mc_with_${VTGenerator}_gen_vt_${Model}.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/LLaVA-VT/eval/pope/pope.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/pope_with_${VTGenerator}_gen_vt_${Model}.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/LLaVA-VT/eval/vizwiz/vizwiz.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/vizwiz_with_${VTGenerator}_gen_vt_${Model}.txt

CUDA_VISIBLE_DEVICES=0 bash scripts/LLaVA-VT/eval/scienceqa/scienceqa.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/scienceqa_with_${VTGenerator}_gen_vt_${Model}.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/LLaVA-VT/eval/gqa/gqa.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/gqa_full_with_${VTGenerator}_gen_vt_${Model}.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/LLaVA-VT/eval/vqav2/vqav2_dev.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/vqav2_dev_with_${VTGenerator}_gen_vt_${Model}.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/LLaVA-VT/eval/textvqa/textvqa.sh ${VTGenerator} ${Model} 2>&1 | tee -a scripts/log/eval_multi_datasets_with_VT/textvqa_with_${VTGenerator}_gen_vt_${Model}.txt

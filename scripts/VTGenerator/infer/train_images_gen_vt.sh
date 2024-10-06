#!/bin/bash

# note: please merge vg_images in one folder for quick inference
# cp -r /path/to/VG_100K/* /path/to/vg_images
# cp -r /path/to/VG_100K_2/* /path/to/vg_images

# infer VTGenerator-13B on llava_instruct_mix665k
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/VTGenerator/infer/train_images_gen_vt/llava_instruct_mix665k_coco_gen_vt.sh VTGenerator-13B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/VTGenerator/infer/train_images_gen_vt/llava_instruct_mix665k_ocrvqa_gen_vt.sh VTGenerator-13B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/VTGenerator/infer/train_images_gen_vt/llava_instruct_mix665k_textcap_gen_vt.sh VTGenerator-13B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/VTGenerator/infer/train_images_gen_vt/llava_instruct_mix665k_vg_gen_vt.sh VTGenerator-13B 

# merge the inference results & store VTGenerator-13B_VT_292k.json & store llava_instruct_mix665k_with_VT.json
python ./scripts/VTGenerator/infer/train_images_gen_vt/merge_llava_instruct_mix665k_all_gen_vt.py \
    --gen_VT_path './playground/data_VT/train_images_gen_vt/VTGenerator-13B_VT_292k.json' \
    --llava_instruct_mix665k_path '/path/to/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json' \
    --image_path './playground/data' \
    --llava_instruct_mix665k_with_VT './playground/data_VT/train_LLaVA-VT/llava_instruct_mix665k_with_VT.json' 
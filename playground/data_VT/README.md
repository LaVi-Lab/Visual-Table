# Data Structure

```
./playground/data_VT
├── eval
│   ├── gqa
│   │   ├── answers
│   │   │   └── gqa_with_VTGenerator-13B_gen_vt
│   │   │       └── LLaVA-VT-13B
│   │   │           ├── merge.jsonl
│   │   │           └── testdev_balanced_predictions.json
│   │   ├── data
│   │   └── gqa_with_VTGenerator-13B_gen_vt.jsonl
│   ├── llava-bench-in-the-wild
│   │   ├── answers
│   │   │   └── llavabench_with_VTGenerator-13B_gen_vt
│   │   │       └── LLaVA-VT-13B
│   │   │           └── LLaVA-VT-13B.jsonl
│   │   ├── llavabench_with_VTGenerator-13B_gen_vt.jsonl
│   │   └── reviews
│   │       └── llavabench_with_VTGenerator-13B_gen_vt
│   │           └── LLaVA-VT-13B
│   │               └── LLaVA-VT-13B_gpt-3.5-turbo-1106.jsonl
│   ├── mmbench
│   │   ├── answers
│   │   │   └── mmbench_with_VTGenerator-13B_gen_vt
│   │   │       └── LLaVA-VT-13B
│   │   │           └── LLaVA-VT-13B.jsonl
│   │   └── answers_upload
│   │       └── mmbench_with_VTGenerator-13B_gen_vt
│   │           └── LLaVA-VT-13B
│   │               └── LLaVA-VT-13B.xlsx
│   ├── mmmu
│   │   ├── answers
│   │   │   └── mmmu_with_VTGenerator-13B_gen_vt
│   │   │       └── LLaVA-VT-13B
│   │   │           ├── gpt_eval_gpt-3.5-turbo-1106
│   │   │           └── mmmu_with_VTGenerator-13B_gen_vt_LLaVA-VT-13B.jsonl
│   │   ├── mmmu.jsonl
│   │   └── mmmu_with_VTGenerator-13B_gen_vt.jsonl
│   ├── mm-vet
│   │   ├── mmvet_with_VTGenerator-13B_gen_vt
│   │   │   ├── answers
│   │   │   │   └── LLaVA-VT-13B
│   │   │   │       └── mmvet_with_VTGenerator-13B_gen_vt_LLaVA-VT-13B.jsonl
│   │   │   └── results
│   │   │       └── LLaVA-VT-13B
│   │   │           ├── mmvet_with_VTGenerator-13B_gen_vt_LLaVA-VT-13B_gpt-4-32k-0613-cap-int-score-1runs.csv
│   │   │           ├── mmvet_with_VTGenerator-13B_gen_vt_LLaVA-VT-13B_gpt-4-32k-0613-cap-score-1runs.csv
│   │   │           ├── mmvet_with_VTGenerator-13B_gen_vt_LLaVA-VT-13B_gpt-4-32k-0613-grade-1runs.json
│   │   │           └── mmvet_with_VTGenerator-13B_gen_vt_LLaVA-VT-13B.json
│   │   └── mmvet_with_VTGenerator-13B_gen_vt.jsonl
│   ├── mmvp_mc
│   │   ├── answers
│   │   │   └── mmvp_mc_with_VTGenerator-13B_gen_vt
│   │   │       └── LLaVA-VT-13B
│   │   │           └── mmvp_mc_with_VTGenerator-13B_gen_vt_LLaVA-VT-13B.jsonl
│   │   ├── mmvp_mc.jsonl
│   │   └── mmvp_mc_with_VTGenerator-13B_gen_vt.jsonl
│   ├── pope
│   │   ├── answers
│   │   │   └── pope_with_VTGenerator-13B_gen_vt
│   │   │       └── LLaVA-VT-13B
│   │   │           └── merge.jsonl
│   │   ├── pope_coco_commitID_e3e39262c85a6a83f26cf5094022a782cb0df58d
│   │   │   ├── coco_pope_adversarial.json
│   │   │   ├── coco_pope_popular.json
│   │   │   └── coco_pope_random.json
│   │   └── pope_with_VTGenerator-13B_gen_vt.jsonl
│   ├── scienceqa
│   │   ├── answers
│   │   │   └── scienceqa_with_VTGenerator-13B_gen_vt
│   │   │       └── LLaVA-VT-13B
│   │   │           ├── LLaVA-VT-13B.jsonl
│   │   │           ├── LLaVA-VT-13B_output.jsonl
│   │   │           └── LLaVA-VT-13B_result.json
│   │   └── scienceqa_with_VTGenerator-13B_gen_vt.json
│   ├── textvqa
│   │   ├── answers
│   │   │   └── textvqa_with_VTGenerator-13B_gen_vt
│   │   │       └── LLaVA-VT-13B
│   │   │           └── merge.jsonl
│   │   └── textvqa_with_VTGenerator-13B_gen_vt.jsonl
│   ├── vizwiz
│   │   ├── answers
│   │   │   └── vizwiz_with_VTGenerator-13B_gen_vt
│   │   │       └── LLaVA-VT-13B
│   │   │           └── merge.jsonl
│   │   ├── answers_upload
│   │   │   └── vizwiz_with_VTGenerator-13B_gen_vt
│   │   │       └── LLaVA-VT-13B
│   │   │           └── LLaVA-VT-13B.json
│   │   └── vizwiz_with_VTGenerator-13B_gen_vt.jsonl
│   └── vqav2
│       ├── answers_upload
│       │   └── vqav2_dev_with_VTGenerator-13B_gen_vt
│       │       └── LLaVA-VT-13B.json
│       └── vqav2_dev_with_VTGenerator-13B_gen_vt.jsonl
├── eval_images_gen_vt
│   ├── gqa_gen_vt
│   │   └── VTGenerator-13B
│   │       └── merge.jsonl
│   ├── gqa_gen_vt.jsonl
│   ├── llavabench_gen_vt
│   │   └── VTGenerator-13B
│   │       └── merge.jsonl
│   ├── llavabench_gen_vt.jsonl
│   ├── mmbench_gen_vt
│   │   └── VTGenerator-13B
│   │       └── merge.jsonl
│   ├── mmbench_gen_vt.jsonl
│   ├── mmmu_gen_vt
│   │   └── VTGenerator-13B
│   │       └── merge.jsonl
│   ├── mmmu_gen_vt.jsonl
│   ├── mmvet_gen_vt
│   │   └── VTGenerator-13B
│   │       └── merge.jsonl
│   ├── mmvet_gen_vt.jsonl
│   ├── mmvp_gen_vt
│   │   └── VTGenerator-13B
│   │       └── merge.jsonl
│   ├── mmvp_gen_vt.jsonl
│   ├── pope_gen_vt
│   │   └── VTGenerator-13B
│   │       └── merge.jsonl
│   ├── pope_gen_vt.jsonl
│   ├── scienceqa_gen_vt
│   │   └── VTGenerator-13B
│   │       └── merge.jsonl
│   ├── scienceqa_gen_vt.jsonl
│   ├── textvqa_gen_vt
│   │   └── VTGenerator-13B
│   │       └── merge.jsonl
│   ├── textvqa_gen_vt.jsonl
│   ├── vizwiz_gen_vt
│   │   └── VTGenerator-13B
│   │       └── merge.jsonl
│   ├── vizwiz_gen_vt.jsonl
│   ├── vqav2_gen_vt
│   │   └── VTGenerator-13B
│   │       └── merge.jsonl
│   └── vqav2_gen_vt.jsonl
├── gpt_eval
│   ├── gqa
│   │   ├── gqa.jsonl
│   │   └── gqa_with_gpt4v_vt.jsonl
│   ├── mmmu
│   │   ├── mmmu.jsonl
│   │   └── mmmu_with_gpt4v_vt.jsonl
│   ├── mmvet
│   │   ├── mmvet.jsonl
│   │   └── mmvet_with_gpt4v_vt.jsonl
│   └── mmvp
│       ├── mmvp.jsonl
│       └── mmvp_with_gpt4v_vt.jsonl
├── README.md
├── train_images_gen_vt
│   ├── llava_instruct_mix665k_coco_gen_vt.jsonl
│   ├── llava_instruct_mix665k_ocrvqa_gen_vt.jsonl
│   ├── llava_instruct_mix665k_textcap_gen_vt.jsonl
│   ├── llava_instruct_mix665k_vg_gen_vt.jsonl
│   └── VTGenerator-13B_VT_292k.json
├── train_LLaVA-VT
│   └── llava_instruct_mix665k_with_VT.json
└── train_VTGenerator
    ├── finetune_VTGenerator_gpt4v_VT_61k.json
    └── pretrain_VTGenerator_llava_instruct_mix199k.json
```
export MAX_INFER_BATCH_SIZE=4

python generate.py \
    --pretrained_model_name_or_path <path-to-fine-tuned-model> \
    --image_dir "images/sdd_coco30k" \
    --prompt_path "prompts/coco_30k.csv" \
    --num_images_per_prompt 1 \
    --use_fp16 \
    --device "cuda:0"
python train_esd.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --removing_concepts \
        "nudity" \
    --validation_prompts \
        "japan body" \
    --num_images_per_prompt 10 \
    --seed 42 \
    --devices 0 0
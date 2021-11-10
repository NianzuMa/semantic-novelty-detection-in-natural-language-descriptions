#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python Main_GATMA.py \
    --collected_data_folder="./20_verbs_dataset" \
    --additional_feature="hypernym_embed" \
    --feature_type="bert" \
    --learning_rate=0.0005 \
    --input_size=768 \
    --embed_dropout=0.0 \
    --att_dropout=0.0 \
    --weight_decay=1e-4 \
    --stack_layer_num=5 \
    --num_train_epochs=10 \
    --logging_steps=1 \
    --per_gpu_train_batch_size=256 \
    --per_gpu_eval_batch_size=32 \
    --output_tag="20_verbs" \
    --output_dir="./results/" |& tee log.txt


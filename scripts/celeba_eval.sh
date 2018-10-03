#!/usr/bin/env bash
device=7
folder=log/celeba/20181002/oracle_way2_shot1_query15_hdim64
for test_way in 2
do
    for test_shot in 5
    do
        CUDA_VISIBLE_DEVICES=${device} python scripts/predict/few_shot/run_eval.py \
            --model.model_path ${folder}/best_model.pt \
            --data.test_way ${test_way} \
            --data.test_shot ${test_shot} \
            --data.test_query 5 \
            --data.test_episodes 1000
    done
done
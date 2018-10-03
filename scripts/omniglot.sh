#!/usr/bin/env bash
dataset=omniglot
date=20181002
way=20
shot=1
encoder=acai
clusters=500
partitions=100
hdim=32
train_mode=ground_truth
python scripts/train/few_shot/run_train.py \
    --data.dataset ${dataset} \
    --data.way ${way} \
    --data.shot ${shot} \
    --data.query 5 \
    --data.test_way 5 \
    --data.test_shot 1 \
    --data.test_query 5 \
    --data.train_episodes 100 \
    --data.test_episodes 100 \
    --data.cuda \
    --data.encoder ${encoder} \
    --data.train_mode ${train_mode} \
    --data.test_mode ground_truth \
    --data.clusters ${clusters} \
    --data.partitions ${partitions} \
    --model.x_dim 1,28,28 \
    --model.hid_dim ${hdim} \
    --train.epochs 300 \
    --log.exp_dir log \
    --log.date ${date}

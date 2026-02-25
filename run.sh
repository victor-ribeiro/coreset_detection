#!/bin/bash

# model=SGDClassifier
model=DecisionTreeClassifier

runs=30

# === Fase 1: Selecao de coresets ===
for dataset in covtype; do
    for frac in 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3; do

        python3 main.py select-coreset \
            --dataset $dataset --method freddy \
            --train_frac $frac --runs $runs \
            --batch_size 512 --alpha .1

        python3 main.py select-coreset \
           --dataset $dataset --method random \
           --train_frac $frac --runs $runs

        python3 main.py select-coreset \
       	    --dataset $dataset --method craig \
            --train_frac $frac --runs $runs

    done
done

# === Fase 2: Treinamento de modelos ===
for dataset in covtype; do
    for frac in 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3; do
        for method in freddy random craig; do
            python3 main.py model-train \
                --model $model \
                --coreset_dir coreset/$dataset/$method/$frac
        done
    done
done

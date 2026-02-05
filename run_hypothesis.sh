#!/bin/bash

# Configurações
model=XGBClassifier
experiment=default_experiment
baseline=random_sampler
alpha=0.05

# Executar testes de hipótese para cada dataset
for dataset in covtype; # adult hepmass higgs;
do
    echo "=========================================="
    echo "Dataset: $dataset"
    echo "=========================================="

    python3 hypothesis_test.py \
        --experiment $experiment \
        --model $model \
        --dataset $dataset \
        --baseline $baseline \
        --alpha $alpha

    echo ""
done

# Para executar com métrica específica:
# python3 hypothesis_test.py --experiment default_experiment --model XGBClassifier --dataset covtype --baseline random --metric accuracy_score

# Para executar com fração específica:
# python3 hypothesis_test.py --experiment default_experiment --model XGBClassifier --dataset covtype --baseline random --frac 0.1

#!/bin/bash

# model=SGDClassifier
# model=DecisionTreeClassifier
# model=RandomForestClassifier
dataset=covtype

runs=10

# === Fase 1: Selecao de coresets ===

for dataset in $dataset; do
    for frac in 0.01 0.02 0.03 0.1; do

          python3 main.py select-coreset \
              --dataset $dataset --method freddy \
              --train_frac $frac --runs $runs \
              --batch_size 512 --alpha .1

        python3 main.py select-coreset \
            --dataset $dataset --method random \
            --train_frac $frac --runs $runs

         # python3 main.py select-coreset \
       	 #    --dataset $dataset --method gradmatch \
         #    --train_frac $frac --runs $runs

        # python3 main.py select-coreset \
       	#    --dataset $dataset --method craig \
        #    --train_frac $frac --runs $runs

     done
 done

# === Fase 2: Treinamento de modelos ===
for model in RandomForestClassifier LogisticRegression XGBClassifier DecisionTreeClassifier SGDClassifier ;do
    for frac in 0.01 0.02 0.03 0.1; do
        for method in freddy random; do
            python3 main.py model-train --model $model --coreset_dir coreset/$dataset/$method/$frac --dataset $dataset
        done
    done
done

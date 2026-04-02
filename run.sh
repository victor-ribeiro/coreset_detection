#!/bin/bash

# model=SGDClassifier
# model=DecisionTreeClassifier
# model=RandomForestClassifier
# dataset=adult covtype

runs=30

# === Fase 1: Selecao de coresets ===

for dataset in covtype; do
    for frac in 0.01 0.05 0.1 0.5 0.75 0.9; do

          python3 main.py select-coreset \
              --dataset $dataset --method freddy \
              --train_frac $frac --runs $runs \
              --batch_size 512 --alpha .15

        #  python3 main.py select-coreset \
        #      --dataset $dataset --method random \
        #      --train_frac $frac --runs $runs

        #    python3 main.py select-coreset \
        # 	     --dataset $dataset --method gradmatch \
        #       --train_frac $frac --runs $runs

    #     # python3 main.py select-coreset \
    #    	#    --dataset $dataset --method craig \
    #     #    --train_frac $frac --runs $runs

     done

# === Fase 1.5: Calculo de coverage post-hoc ===
    for frac in 0.01 0.05 0.1 0.5 0.75 0.9; do
    # for frac in  0.1 0.5 0.75 0.9; do

        python3 main.py compute-coverage \
            --dataset $dataset \
            --coreset_dir coreset/$dataset/freddy/$frac

        # python3 main.py compute-coverage \
        #     --dataset $dataset \
        #     --coreset_dir coreset/$dataset/gradmatch/$frac

    	# python3 main.py compute-coverage \
        #     --dataset $dataset \
        #     --coreset_dir coreset/$dataset/random/$frac


    done

    #  python main.py select-coreset --dataset $dataset --method none --train_frac 1.0 --runs $runs

# === Fase 2: Treinamento de modelos ===
    for model in RandomForestClassifier LogisticRegression XGBClassifier DecisionTreeClassifier SGDClassifier ;do
        for frac in  0.01 0.05 0.1 0.5 0.75 0.9; do
            for method in freddy; do
                python3 main.py model-train --model $model --coreset_dir coreset/$dataset/$method/$frac --dataset $dataset
            done
        done
    done
 done

# !/bin/bash

# # XGBRegressor
# predictmds
# sgemm
# storage_perf

# # XGBClassifier
# adult
# covtype
# hepmass
# higgs


# dataset=storage_perf
model=XGBRegressor


for dataset in predictmds sgemm storage_perf:
do
    for frac in 100 1000 10000 100000;
    do
        for method in freddy random gradmatch craig;
        do 
            python main.py --dataset $dataset --method $method  --model $model --run 6 --tol .001 --resample 5 --train_frac $frac --batch_size 500 --alpha 1 --name mag_order
        done
    done
    python main.py --dataset $dataset --method none  --model $model --run 6 --resample 5
done


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


# model=XGBRegressor
# model=RandomForestClassifier
model=XGBClassifier
# name=alpha
name=mag_order

for dataset in covtype;
do
    for frac in 1000 10000 100000;
    # for alpha in .1 .5 1 2 5 ;
    # for alpha in  2 2.5 5;
    do
        # for method in freddy gradmatch random;
        for method in freddy random;
        do 
            # python main.py --dataset $dataset --method $method  --model $model --run 6 --tol .001 --resample 5 --train_frac 10000 --batch_size 64 --alpha $alpha --name $name
            python main.py --dataset $dataset --method $method  --model $model --run 6 --tol .001 --resample 5 --train_frac $frac --batch_size 1024 --alpha 1 --name $name
        done
    done
    # python main.py --dataset $dataset --method none  --model $model --run 6 --resample 5 --name mag_order
done

# 
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


# model=RandomForestClassifier
model=XGBClassifier
# model=XGBRegressor
# name=alpha
name=default_experiment

for dataset in covtype adult;
do
    for frac in .1 .2 .3 .4 .5;
    # for alpha in .001 .005 .01 .05 .1 .5 2;
    do
        # for method in freddy gradmatch random;
        for method in freddy;
        do 
            # python main.py --dataset $dataset --method $method  --model $model --run 10 --tol .001 --resample 3 --train_frac 10000 --batch_size 500 --alpha $alpha --name $name
            python main.py --dataset $dataset --method $method  --model $model --run 3 --tol .00001 --resample 10 --train_frac $frac --batch_size 1024 --alpha .01 --name $name
        done
    done
    # python main.py --dataset $dataset --method none  --model $model --run 6 --resample 5 --name $name
done

# 

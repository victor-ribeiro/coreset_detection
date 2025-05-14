# !/bin/bash

dataset=storage_perf
model=XGBRegressor

# dataset=covtype
# model=XGBClassifier

for frac in .1 .2 .3 .4;
do
    for method in freddy;
    # for method in gradmatch random;
    do 
        python main.py --dataset $dataset --method $method  --model $model --run 6 --tol .001 --resample 5 --train_frac $frac --batch_size 500 --alpha 1
    done
done


# python main.py --dataset $dataset --method none  --model $model --run 10 --resample 5 --train_frac 1

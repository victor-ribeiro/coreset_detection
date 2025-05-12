# !/bin/bash

# dataset=sgemm
# model=XGBRegressor

dataset=covtype
model=XGBClassifier

for frac in .1 .2 .3 .4 ;
do
    # for method in pmi_kmeans freddy random craig;
    for method in freddy;
    do 
        python main.py --dataset $dataset --method $method  --model $model --run 10 --tol .001 --resample 5 --train_frac $frac --batch_size 500 --alpha .5
    done
done


# python main.py --dataset $dataset --method none  --model $model --run 10 --resample 5 --train_frac 1

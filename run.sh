# !/bin/bash

dataset=adult
model=XGBClassifier
# dataset=adult
# model=XGBClassifier

for frac in .1  .2 .3 .4 .5 .6;
do
    # for method in pmi_kmeans freddy random craig;
    for method in craig;
    do 
        python main.py --dataset $dataset --method $method  --model $model --run 10 --tol .1 --resample 5 --train_frac $frac --batch_size 500
    done
done


# python main.py --dataset $dataset --method none  --model $model --run 10 --resample 5 --train_frac 1

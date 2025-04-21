# !/bin/bash

dataset=adult
model=XGBClassifier
# dataset=adult
# model=XGBClassifier

for frac in .1 .2 .3 .4 .5 .6 .7 .8 .9;
do
    for method in craig ;  
    do 
        python main.py --dataset $dataset --method $method  --model $model --run 10 --resample 5 --train_frac $frac 
    done
done


python main.py --dataset $dataset --method none  --model $model --run 10 --resample 5 --train_frac 1

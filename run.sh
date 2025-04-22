# !/bin/bash

dataset=adult
model=XGBClassifier
# dataset=adult
# model=XGBClassifier

for method in pmi_kmeans;  
do
   for frac in .1  .2 .3 .4 .5  .6 .7 .8 .9;
    do 
        python main.py --dataset $dataset --method $method  --model $model --run 10 --resample 5 --train_frac $frac 
    done
done


# python main.py --dataset $dataset --method none  --model $model --run 10 --resample 5 --train_frac 1

# !/bin/bash


for frac in .1 .2 .3 .4 .5 .6 .7 .8 .9;
do
    # for method in pmi_kmeans freddy craig none; 
    for method in pmi_kmeans none; 
    do 
        python main.py --dataset covtype --method $method  --model XGBClassifier --run 10 --resample 5 --train_frac $frac 
    done
done

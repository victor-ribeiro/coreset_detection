# !/bin/bash


for frac in .1 .2 .3 .4 .5 .6 .7 .8 .9;
do
    for method in craig pmi_kmeans freddy; 
    # for method in pmi_kmeans none; 
    do 
        python main.py --dataset bike_share --method $method  --model DecisionTreeRegressor --run 10 --resample 5 --train_frac $frac 
    done
done

python main.py --dataset covtype --method $none  --model XGBClassifier --run 10 --resample 5 --train_frac 1

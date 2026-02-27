# # !/bin/bash

# conda run -n coreset_detection python main.py --dataset adult --method freddy  --model XGBClassifier --tol .001 --train_frac .05 --batch_size 500


# # # XGBRegressor
# # predictmds
# # sgemm
# # storage_perf

# # # XGBClassifier
# # adult
# # covtype
# # hepmass
# # higgs


model=SGDClassifier
#model=XGBClassifier
# model=XGBRegressor
# name=default_experiment

name=alpha_xp
for dataset in covtype;
do
   for alpha in .05 .1 .2 .25 .5 .75 1 2;
   do
     for method in freddy;
       do
           python3 main.py --dataset $dataset --method $method  --model $model --run 10 --tol .001 --resample 1 --train_frac .05 --batch_size 2048 --name $name --alpha $alpha
       done
    done
done 

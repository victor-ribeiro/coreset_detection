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


# model=RandomForestClassifier
model=XGBClassifier
# model=XGBRegressor
# name=alpha
name=default_experiment

 for dataset in covtype;
 do
    for frac in .01 .02 .03 .04 0.05 .1 .2 .3 .4 .5 .6 .7 .8 .9;
    do
    #    for method in freddy random gradmatch;
       for method in random;
       do
           python3 main.py --dataset $dataset --method $method  --model $model --run 10 --resample 10 --tol .001  --train_frac $frac --batch_size 1024 --name $name --alpha .1
        #    python3 main.py --dataset $dataset --method $method  --model $model --run 10 --tol .001 --resample 1 --train_frac .1 --batch_size 2048 --name $name --alpha $alpha
       done
    done
done
# valgrind --leak-check=yes python3.13 main.py --dataset adult --method freddy  --model $model --run 1 --resample 1 --train_frac .1 --batch_size 500 --name $name &
# valgrind --show-possibly-lost=no --leak-check=full --show-leak-kinds=all hpcrun python main.py --dataset adult --method freddy  --model $model --run 1 --resample 1 --train_frac .1 --batch_size 500 --name $name &

# if [ -f mem_log.log ]; then
#     rm mem_log.log
# fi

# valgrind --show-possibly-lost=no --leak-check=full --show-leak-kinds=all python main.py --dataset adult --method freddy  --model $model --run 1 --resample 1 --train_frac .1 --batch_size 500 --name $name &> mem_log.log &

#hpcrun -a python --disable-auditor -e GA python main.py --dataset adult --method freddy  --model $model --run 1 --resample 1 --train_frac .1 --batch_size 500 --name $name


#done

# 

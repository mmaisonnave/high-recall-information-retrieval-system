TARGET_RECALL=0.8
PROPORTION=1.0
SCRIPT=scal_simulation.py
<<<<<<< HEAD
for RF in random 
	#half_relevance_half_uncertainty relevance uncertainty 1quarter_relevance_3quarters_uncertainty 3quarter_relevance_1quarters_uncertainty random
do
    for N in 50 100 250 500 1000 2000 3000 4500 6000
    do
        for I in {1..5}
        do
            RANDOM_NUMBER=$RANDOM
            echo python $SCRIPT --N=$N --target-recall=$TARGET_RECALL --seed=$RANDOM_NUMBER --proportion-relevance=$PROPORTION --ranking-function=$RF --sbert
            python $SCRIPT --N=$N --target-recall=$TARGET_RECALL --seed=$RANDOM_NUMBER --proportion-relevance=$PROPORTION --ranking-function=$RF --sbert
=======
# python $SCRIPT --N=100 --target-recall=0.8 --seed=$RANDOM --proportion-relevance=1.0 --ranking-function=min_distance
# python $SCRIPT --N=250 --target-recall=0.8 --seed=$RANDOM --proportion-relevance=1.0 --ranking-function=min_distance
# python $SCRIPT --N=500 --target-recall=0.8 --seed=$RANDOM --proportion-relevance=1.0 --ranking-function=min_distance
# 
# python $SCRIPT --N=500 --target-recall=0.8 --seed=$RANDOM --proportion-relevance=1.0 --ranking-function=relevance
# 
# python $SCRIPT --N=1000 --target-recall=0.8 --seed=$RANDOM --proportion-relevance=1.0 --ranking-function=min_distance
# 
# python $SCRIPT --N=1000 --target-recall=0.8 --seed=$RANDOM --proportion-relevance=1.0 --ranking-function=relevance
# 
# python $SCRIPT --N=1000 --target-recall=0.8 --seed=$RANDOM --proportion-relevance=1.0 --ranking-function=relevance
# 
# python $SCRIPT --N=2000 --target-recall=0.8 --seed=$RANDOM --proportion-relevance=1.0 --ranking-function=avg_distance
# 
# 
# python $SCRIPT --N=2000 --target-recall=0.8 --seed=$RANDOM --proportion-relevance=1.0 --ranking-function=min_distance
# python $SCRIPT --N=6000 --target-recall=0.8 --seed=$RANDOM --proportion-relevance=1.0 --ranking-function=min_distance

# for RF in uncertainty relevance 
SCRIPT=scal_simulation.py
for RF in uncertainty 1quarter_relevance_3quarters_uncertainty 3quarter_relevance_1quarters_uncertainty
# for RF in  uncertainty 
do
    for N in 50 100 250 500 1000 2000 3000 4500 6000
    do
        for I in {1..10}
        do
            RANDOM_NUMBER=$RANDOM
            echo python $SCRIPT --N=$N --target-recall=$TARGET_RECALL --seed=$RANDOM_NUMBER --proportion-relevance=$PROPORTION --ranking-function=$RF --glove
            python $SCRIPT --N=$N --target-recall=$TARGET_RECALL --seed=$RANDOM_NUMBER --proportion-relevance=$PROPORTION --ranking-function=$RF --glove
>>>>>>> 52e911f4220bf7f2382e262cc92b6368c417043b
        done
    done
done

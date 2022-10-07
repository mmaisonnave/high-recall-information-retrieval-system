# TARGET_RECALL=0.8
# for PROPORTION in 0\.0 0\.25 0\.50 0\.75 1\.0
# do
#     for N in 50 100 250 500 1000 2000 3000 4500 6000
#     do
#         for I in {1..10}
#         do
#             RANDOM_NUMBER=$RANDOM
#             echo python scal_simulation.py --N=$N --target-recall=$TARGET_RECALL --seed=$RANDOM_NUMBER --proportion-relevance=$PROPORTION
#             python scal_simulation.py --N=$N --target-recall=$TARGET_RECALL --seed=$RANDOM_NUMBER --proportion-relevance=$PROPORTION
#         done
#    done
# done


TARGET_RECALL=0.8
PROPORTION=1.0
SCRIPT=scal_simulation.py
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
        done
    done
done

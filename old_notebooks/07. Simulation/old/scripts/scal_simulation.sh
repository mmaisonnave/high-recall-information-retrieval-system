TARGET_RECALL=0.8
PROPORTION=1.0
SCRIPT=scal_simulation.py
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
        done
    done
done

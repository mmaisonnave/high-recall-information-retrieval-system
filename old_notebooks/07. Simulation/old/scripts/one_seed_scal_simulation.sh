TARGET_RECALL=0.8
for PROPORTION in 0\.0 0\.25 0\.50 0\.75 1\.0
# 0\.0 0\.25 0\.50 0\.75 1\.0
do
    for N in 50 100 250 500 1000 2000 3000 4500 6000
    do
        for I in 1 2 3 4 5 
        do
            RANDOM_NUMBER=$RANDOM
            echo python scal_simulation.py --N=$N --target-recall=$TARGET_RECALL --seed=$RANDOM_NUMBER --proportion-relevance=$PROPORTION 
            python scal_simulation.py --N=$N --target-recall=$TARGET_RECALL --seed=$RANDOM_NUMBER --proportion-relevance=$PROPORTION 
	    #--diversity --average-diversity
        done
    done
done

TARGET_RECALL=0.8
for N in 50 100 250 500 1000 2000 3000 4500 6000
do
    for I in {1..40}
    do
        RANDOM_NUMBER=$RANDOM
        echo python scal_simulation.py --N=$N --target-recall=$TARGET_RECALL --seed=$RANDOM_NUMBER
        python scal_simulation.py --N=$N --target-recall=$TARGET_RECALL --seed=$RANDOM_NUMBER
    done
done

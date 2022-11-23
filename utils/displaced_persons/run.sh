PYTHON_SCRIPT=simulation.py

MODEL_TYPE=logreg
REPRESENTATION=bow
# %5 10% %20 %25 %50 %75
for seed in {1..5}
do
    for N in 364 728 1456 1820 3641 5461 7282 
    do
        for n in 1 3 5 10 20
        do
            echo python $PYTHON_SCRIPT --N $N --target-recall 0.8 --n $n --seed $RANDOM \
                                 --ranking-function relevance \
                                --model-type $MODEL_TYPE --representation $REPRESENTATION
            python $PYTHON_SCRIPT --N $N --target-recall 0.8 --n $n --seed $RANDOM \
                                --ranking-function relevance \
                                --model-type $MODEL_TYPE --representation $REPRESENTATION
        done
    done
done

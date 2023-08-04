TARGET_RECALL=0.8
RANKING_FUNCTION=relevance
REPRESENTATION=BoW
PYTHON_SCRIPT=scal_simulation_with_heatmap.py
PYTHON_INTERPRETER=python
for N in 50 100 250 500 1000 2000 3000 4500 6000
do
    for n in 1 2 3 4 5 7 10 20
    do
        for repetition in {1..7}
        do
            SEED=$RANDOM
            echo $PYTHON_INTERPRETER $PYTHON_SCRIPT --N $N --n $n --ranking-function $RANKING_FUNCTION --target-recall $TARGET_RECALL --representation $REPRESENTATION --seed $SEED 
            $PYTHON_INTERPRETER $PYTHON_SCRIPT --N $N --n $n --ranking-function $RANKING_FUNCTION --target-recall $TARGET_RECALL --representation $REPRESENTATION --seed $SEED 
        done
    done
done

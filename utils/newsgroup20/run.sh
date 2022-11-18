PYTHON_SCRIPT=simulation.py

MODEL_TYPE=logreg
REPRESENTATION=bow
# %5 10% %20 %25 %50 %75
for N in 941 1882 3761 4707 9414 14121
do
    for n in 1 3 5 10 20
    do
        for file in $( ls /home/ec2-user/SageMaker/mariano/datasets/20news-18828/files/ )
        do
            echo python $PYTHON_SCRIPT --N $N --target-recall 0.8 --n $n --seed $RANDOM \
                                --category $file --ranking-function relevance \
                                --model-type $MODEL_TYPE --representation $REPRESENTATION
            python $PYTHON_SCRIPT --N $N --target-recall 0.8 --n $n --seed $RANDOM \
                                --category $file --ranking-function relevance \
                                --model-type $MODEL_TYPE --representation $REPRESENTATION
        done
    done
done
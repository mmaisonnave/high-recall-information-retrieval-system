PYTHON_SCRIPT=simulation.py

MODEL_TYPE=logreg
REPRESENTATION=bow
N=18828
n=20
# %5 10% %20 %25 %50 %75
# 
for MODEL_TYPE in logreg svm
do
	for REPRESENTATION in bow glove sbert
	do
		for file in $( ls /home/ec2-user/SageMaker/mariano/datasets/20news-18828/files/ )
		do
			for seed in {1..5}
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
done

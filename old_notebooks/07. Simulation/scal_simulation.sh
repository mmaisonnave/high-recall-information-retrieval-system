for N in 50 250 500 1000 2000 3000 4000 5000 6000
do
    for TARGET_RECALL in 0.6 0.7 0.8 0.9
    do
        echo python scal_simulation.py N=$N target_recall=$TARGET_RECALL
        python scal_simulation.py N=$N target_recall=$TARGET_RECALL
    done
done
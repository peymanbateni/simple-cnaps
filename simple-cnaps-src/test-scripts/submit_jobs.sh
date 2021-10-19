for ways in 5 10
    do
        for shots in 1 5
	do
	    for feature_adaptation in film
            do
	        for dataset in mini tiered
		do
		    sbatch --job-name=${ways}_${shots}_${dataset}_simple --output=mt_${model}_${feature_adaptation}_${shots}_shots_${ways}_ways_on_${dataset}_dataset.out train_simple_cnaps_mt.sh ${feature_adaptation} ${shots} ${ways} ${dataset}
		done
            done
	done
    done

This code replicates an experiment thought up by Vincent.

This experiment will:
1. Generate 4 sine wave datasets using different generators and split each dataset in train and test splits. 
2. Train TimeVAE on each dataset, generating 4 synthetic datasets.
3. Evaluate each synthetic dataset with TSGBench. 

We use the interpretable version of TimeVAE with default parameters.
- TimeVAEs parameters can be changed in /timeVAE/test_vae.py.

Make sure to create the following environments before starting the experiment:
- Experiment.sif
- TimeVAE.sif
- TSGBench.sif

The environments can be created using apptainer:
$ apptainer build <FILENAME>.sif <FILENAME>.def

The experiment can be replicated on DIKUs cluster:
$ cd experiment
$ sbatch experiment.sh

Various parameters can be changed via the experiment.sh script.

The above commands and scripts should be changed to reflect your dir structure.
#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=6000M
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=4:00:00
~/install-dir/bin/apptainer run --nv ~/sif_files/Experiment.sif bash -c "cd generate_sines && python generate_sines.py --no 10000 --seq_len 24 --dim 6 --freq_range 1 5 --amp_range 0.1 0.9 --phase_range -3.14 3.14"

~/install-dir/bin/apptainer run --nv ~/sif_files/TimeVAE.sif bash -c "cd timeVAE && python test_vae.py --dataset_name jakob --dataset_state gen"
~/install-dir/bin/apptainer run --nv ~/sif_files/TimeVAE.sif bash -c "cd timeVAE && python test_vae.py --dataset_name timegan --dataset_state gen_1"
~/install-dir/bin/apptainer run --nv ~/sif_files/TimeVAE.sif bash -c "cd timeVAE && python test_vae.py --dataset_name timegan --dataset_state gen_2"
~/install-dir/bin/apptainer run --nv ~/sif_files/TimeVAE.sif bash -c "cd timeVAE && python test_vae.py --dataset_name timegan --dataset_state gen_3"
~/install-dir/bin/apptainer run --nv ~/sif_files/TimeVAE.sif bash -c "cd timeVAE && python test_vae.py --dataset_name timegan --dataset_state gen_4"

~/install-dir/bin/apptainer run --nv ~/sif_files/TSGBench.sif bash -c "cd TSGBench && python eval/ds_ps.py --method_name timevae --dataset_name jakob --dataset_state gen --gpu_id 0 --gpu_fraction 0.99 && python eval/c_fid/c_fid.py --method_name timevae --dataset_name jakob --dataset_state gen --gpu_id 0 && python eval/feature_distance_eval.py --method_name timevae --dataset_name jakob --dataset_state gen --gpu_id 0 && python eval/visualization.py --method_name timevae --dataset_name jakob --dataset_state gen"
~/install-dir/bin/apptainer run --nv ~/sif_files/TSGBench.sif bash -c "cd TSGBench && python eval/ds_ps.py --method_name timevae --dataset_name timegan --dataset_state gen_1 --gpu_id 0 --gpu_fraction 0.99 && python eval/c_fid/c_fid.py --method_name timevae --dataset_name timegan --dataset_state gen_1 --gpu_id 0 && python eval/feature_distance_eval.py --method_name timevae --dataset_name timegan --dataset_state gen_1 --gpu_id 0 && python eval/visualization.py --method_name timevae --dataset_name timegan --dataset_state gen_1"
~/install-dir/bin/apptainer run --nv ~/sif_files/TSGBench.sif bash -c "cd TSGBench && python eval/ds_ps.py --method_name timevae --dataset_name timegan --dataset_state gen_2 --gpu_id 0 --gpu_fraction 0.99 && python eval/c_fid/c_fid.py --method_name timevae --dataset_name timegan --dataset_state gen_2 --gpu_id 0 && python eval/feature_distance_eval.py --method_name timevae --dataset_name timegan --dataset_state gen_2 --gpu_id 0 && python eval/visualization.py --method_name timevae --dataset_name timegan --dataset_state gen_2"
~/install-dir/bin/apptainer run --nv ~/sif_files/TSGBench.sif bash -c "cd TSGBench && python eval/ds_ps.py --method_name timevae --dataset_name timegan --dataset_state gen_3 --gpu_id 0 --gpu_fraction 0.99 && python eval/c_fid/c_fid.py --method_name timevae --dataset_name timegan --dataset_state gen_3 --gpu_id 0 && python eval/feature_distance_eval.py --method_name timevae --dataset_name timegan --dataset_state gen_3 --gpu_id 0 && python eval/visualization.py --method_name timevae --dataset_name timegan --dataset_state gen_3"
~/install-dir/bin/apptainer run --nv ~/sif_files/TSGBench.sif bash -c "cd TSGBench && python eval/ds_ps.py --method_name timevae --dataset_name timegan --dataset_state gen_4 --gpu_id 0 --gpu_fraction 0.99 && python eval/c_fid/c_fid.py --method_name timevae --dataset_name timegan --dataset_state gen_4 --gpu_id 0 && python eval/feature_distance_eval.py --method_name timevae --dataset_name timegan --dataset_state gen_4 --gpu_id 0 && python eval/visualization.py --method_name timevae --dataset_name timegan --dataset_state gen_4"
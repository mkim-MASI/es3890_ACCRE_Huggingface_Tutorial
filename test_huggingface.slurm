#!/bin/bash
#SBATCH --mail-user=michael.kim@vanderbilt.edu  #For sending email when job is complete
#SBATCH --mail-type=FAIL,END        # times you get notified by email
#SBATCH --account=es3890_acc        # Project account name
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   #specifying the type of GPU --gres=gpu:<gpu-type>:<gpu-num>
#SBATCH --partition=batch_gpu      # partition to use for the job submissino 
#SBATCH --nodes=1                  # Request one node
#SBATCH --mem-per-cpu=16G           # Memory per CPU core (e.g., 1GB)
#SBATCH --time=00:05:00            # Maximum runtime (D-HH:MM:SS or HH:MM:SS)

#activate your environment
source /home/kimm58/huggingface_demo/bin/activate

#run your script
python3 /home/kimm58/test_huggingface3.py

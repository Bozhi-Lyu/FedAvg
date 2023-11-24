#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J FedAvg_2NN_test
### -- ask for number of cores (default: 1) --
#BSUB -n 6
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00

#BSUB -J 2NN[1-24]

#BSUB -R "span[ptile=4]"
# request 6GB of system-memory
#BSUB -R "rusage[mem=4GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u email
### -- Specifies to write an email to the user address when the job begins
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o FedAvg_2NN_test_%J_out.txt
#BSUB -e FedAvg_2NN_test_%J_err.txt
# -- end of LSF options --

nvidia-smi

module load gcc/10.3.0-binutils-2.36.1
module load python3/3.9.11
module load openblas/0.3.19
module load numpy/1.22.3-python-3.9.11-openblas-0.3.19
source ../torch_dl/bin/activate

python G${LSB_JOBINDEX}.py

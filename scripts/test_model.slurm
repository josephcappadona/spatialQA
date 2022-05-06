#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
cd $SCRATCH/spatialQA
python scripts/test.py $1
python scripts/analyze.py $1
"

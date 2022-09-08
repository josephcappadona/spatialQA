`ssh greene`

For HPC CPU session: `srun --nodes=1 --cpus-per-task=1 --mem=32GB --time=1:00:00 --pty /bin/bash`
For HPC GPU session: `srun --nodes=1 --cpus-per-task=1 --mem=32GB --time=1:00:00 --gres=gpu:1 --pty /bin/bash`

On Greene:
```
cd ${SCRATCH}
cp /scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz .
gunzip -vvv ./overlay-25GB-500K.ext3.gz
```

On Greene: `srun --nodes=1 --cpus-per-task=1 --mem=32GB --time=1:00:00 --gres=gpu:1 --pty /bin/bash`
In interactive sessions:
```
singularity exec --overlay $SCRATCH/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash

cd /ext3/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
# set prefix to /ext3/miniconda3 when prompted

pip install torch torchvision transformers datasets nlp sklearn ray[tune]==1.11 bayesian-optimization gputil matplotlib

mkdir $SCRATCH/.cache
mkdir $SCRATCH/.conda
rm -rfv .conda
rm -rfv .cache
ln -s $SCRATCH/.conda ./
ln -s $SCRATCH/.cache ./

# now can run python w/ libraries
```
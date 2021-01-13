PYTHONPATH=$PYTHONPATH:../../ \
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
python -u -m main --model deit_tiny_patch16_224 --batch-size 64 --data-path /mnt/lustre/share/images --output_dir ./

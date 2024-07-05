# training
PYTHONPATH=./ srun -p mm_lol --quotatype=reserved --job-name=probe --gres=gpu:8 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=96 --kill-on-bad-exit=1 --pty sh hydit/train_human_animation.sh --multireso --reso-step 16

# inference
srun -p mm_lol --quotatype=spot --job-name=probe --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16 --kill-on-bad-exit=1 --pty python sample_t2i.py --prompt "a beautiful chinese girl is dancing" --no-enhance
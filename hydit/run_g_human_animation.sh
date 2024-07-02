model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "
cuda_devices=$CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed --include localhost:$cuda_devices --master_port 24678 hydit/train_human_animation_stage1.py ${params}  "$@"
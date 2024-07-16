task_flag="ref_net_attn_inject_val_21k_uncond_basehuman_lrdecay"                                # the task flag is used to identify folders.
resume=./ckpts/t2i/model/                                    # checkpoint root for resume
index_file=dataset/porcelain/jsons/porcelain.json            # index file for dataloader
results_dir=./log_EXP                                        # save root for results
batch_size=4                                                 # training batch size
image_size="512 896"                                                  # training image resolution
grad_accu_steps=1                                            # gradient accumulation
warmup_num_steps=0                                           # warm-up steps
lr=0.0001                                                    # learning rate
ckpt_every=2000                                              # create a ckpt every a few steps.
ckpt_latest_every=5000                                       # create a ckpt named `latest.pt` every a few steps.


sh $(dirname "$0")/run_g_human_animation.sh \
    --task-flag ${task_flag} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.03 \
    --predict-type v_prediction \
    --uncond-p 0.44 \
    --uncond-p-t5 0.44 \
    --index-file ${index_file} \
    --random-flip \
    --lr ${lr} \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --use-flash-attn \
    --use-fp16 \
    --ema-dtype fp32 \
    --results-dir ${results_dir} \
    --resume-split \
    --resume ${resume} \
    --ckpt-every ${ckpt_every} \
    --ckpt-latest-every ${ckpt_latest_every} \
    --log-every 10 \
    --deepspeed \
    --deepspeed-optimizer \
    --use-zero-stage 2 \
    --epochs 10 \
    --reso-step 32 \
    --data-config ./dataset/human_animation_yamls/stage1_pose1024.yaml \
    "$@"
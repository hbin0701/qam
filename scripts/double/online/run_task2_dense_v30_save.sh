#!/bin/bash
# Run QAM-EDIT on Double-Play Task 2 with DENSE V30 reward (ONLINE-ONLY)
# Save every eval checkpoint + keep all checkpoints + save eval video/pose artifacts.

# JAX Memory Management
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.20

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH="/rlwrld3/home/hyeonbin/RL/ogbench:${PYTHONPATH:-}"

MUJOCO_GL=egl python main.py \
    --run_group=double_task2_rewards_online_save \
    --project=0219-double-tmp1 \
    --run_name="[Final, Online, SaveAll] Double, V30, QAM" \
    --agent=agents/qam.py \
    --tags=QAM_EDIT,dense_v30,online,save_all \
    --seed=10001 \
    --env_name=cube-double-play-singletask-task2-v0 \
    --cube_success_threshold=0.04 \
    --ogbench_dataset_dir=/rlwrld3/home/hyeonbin/.ogbench/data/cube-double-play-1m \
    --sparse=False \
    --dense_reward_version=v30 \
    --terminal_bonus=20 \
    --dense_shaping_lambda=20 \
    --horizon_length=5 \
    --agent.action_chunking=True \
    --agent.inv_temp=1.0 \
    --agent.fql_alpha=0.0 \
    --agent.edit_scale=0.0 \
    --offline_steps=0 \
    --online_steps=200000 \
    --eval_interval=50000 \
    --log_interval=50000 \
    --dataset_replace_interval=0 \
    --video_episodes=1 \
    --video_frame_skip=1 \
    --save_every_eval=True \
    --save_eval_artifacts=True \
    --auto_cleanup=False


#!/bin/bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=true
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V1=1
export VERL_LOGGING_LEVEL=INFO
export VERL_QUEUE_LOGGING_LEVEL=INFO

export RAY_LOG_TO_DRIVER=1
export RAY_LOGGING_LEVEL=INFO

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/examples/math/runtime_env.yaml"}

KL_LOSS_COEF=0.001
PPO_EPOCHS=1
SEQUENCE_PARALLEL=2
ROLLOUT_TP_SIZE=1
ACTOR_LR=${ACTOR_LR:-1e-6}

LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-mean}
POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-opmd}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-rloo}
PMD_TAU=${PMD_TAU:-0.01}
PMD_ALPHA=${PMD_ALPHA:-1.0}
PMD_REWARD_LB=${PMD_REWARD_LB:-null}
PMD_REWARD_UB=${PMD_REWARD_UB:-null}
RESET_OPTIMIZER_FREQ=${RESET_OPTIMIZER_FREQ:-0}
CRITIC_VALUE_LOSS_TYPE=${CRITIC_VALUE_LOSS_TYPE:-mle}

PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_MODEL_LEN * 2))}
INFER_PPO_MAX_TOKEN_LEN_PER_GPU=${INFER_PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_MODEL_LEN * 3))}

N_NODES=${N_NODES:-1}
PROJECT_NAME=${PROJECT_NAME:-verl_dapo_pmd}
EXP_NAME=qwen2_5_7b_${POLICY_LOSS_MODE}ln_adv${ADV_ESTIMATOR}_stal${MAX_STALENESS}bsz512_mbsz32_n16_ep${PPO_EPOCHS}_t${PMD_TAU}_lr${ACTOR_LR}_ro${RESET_OPTIMIZER_FREQ}

MODEL_PATH="Qwen/Qwen2.5-7B"
TRAIN_DATA="${DATA_DIR}/data/dapo-math-17k.parquet"
VAL_DATA="["${DATA_DIR}/data/aime-2024-repeat1.parquet","${DATA_DIR}/data/aime-2025-repeat1.parquet"]"

ray job submit --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m openkimi.pmd.main_pmd \
    --config-path "${WORKING_DIR}/verl/verl/trainer/config" \
    --config-name ppo_trainer \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    +algorithm.partition_tau=${PMD_TAU} \
    +algorithm.partition_reward_lb=${PMD_REWARD_LB} \
    +algorithm.partition_reward_ub=${PMD_REWARD_UB} \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.use_kl_in_reward=false \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=512 \
    data.max_prompt_length=$PROMPT_LENGTH \
    data.max_response_length=$RESPONSE_LENGTH \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.shuffle=true \
    data.prompt_key=prompt \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.nccl_timeout=3600 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SEQUENCE_PARALLEL \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU} \
    +actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${ACTOR_PPO_INFER_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS_MODE} \
    +actor_rollout_ref.actor.policy_loss.pmd_tau=${PMD_TAU} \
    +actor_rollout_ref.actor.policy_loss.pmd_alpha=${PMD_ALPHA} \
    +actor_rollout_ref.actor.reset_optimizer_states_freq=${RESET_OPTIMIZER_FREQ} \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SEQUENCE_PARALLEL \
    critic.enable=false \
    +critic.value_loss_type=${CRITIC_VALUE_LOSS_TYPE} \
    reward_model.enable=false \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.max_resp_len=8192 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=false \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=128 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    trainer.total_training_steps=3000 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    2>&1 | tee -a $EXP_NAME.log
    \
    $@

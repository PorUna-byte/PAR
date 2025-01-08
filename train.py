# Copyright (c) 2024 Stepfun AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Main script for training.
Remember to allocate enough RAM before running this (you need aroundd 800 GB for Llama-13B).
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import os
import trainers
import wandb
import json
from models.models import AutoModelForCausalLMWithScalarHead, AutoModelForCausalLMWithScalarHeadODIN
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import dataloaders.dataloader as dataloader
import dataloaders.dataset as dataset
from utils.utils import setup_logger, log_message_rank0, setup_distributed, cleanup_distributed, all_gather_if_needed
import deepspeed
from configs.config import get_args
from peft import LoraConfig, get_peft_model

def build_trainer(config, tokenizer: AutoTokenizer, policy_engine, reward_engine, reference_engine, critic_engine):
    """Build the trainer"""
    if config.wandb_enabled and config.global_rank==0:
        os.environ['WANDB_CACHE_DIR'] = config.cache_dir
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            config=config,
            dir=config.cache_dir,
            name=config.exp_name,
        )
    
    if config.loss_name == 'sft':
        train_split = 'train_sft'
        test_split = 'test_prefs'
    else:
        if config.reward_discrete or config.reward_par or config.reward_lsc:
            train_split = config.model_name+'_train_prefs_plus' 
            test_split = config.model_name+'_test_prefs_plus'
        else:
            train_split = 'train_prefs'
            test_split = 'test_prefs'
    
    train_data_loader_class = getattr(dataloader, config.train_dataloader)
    train_iterator = train_data_loader_class(
        config,
        tokenizer,
        split=train_split,
        batch_size=config.global_batch_size,
        n_examples=config.n_examples,
        n_epochs=config.n_epochs,
    )

    test_data_loader_class = getattr(dataloader, config.test_dataloader)
    if test_split != None:
        eval_iterator = test_data_loader_class(
            config,
            tokenizer,
            split=test_split,
            batch_size=config.global_batch_size,
            n_examples=config.n_eval_examples, 
            n_epochs=(1 if config.n_eval_examples is None else None),
        )
    else:
        eval_iterator = None

    TrainerClass = getattr(trainers, config.trainer)
    trainer = TrainerClass(
        config,
        tokenizer,  
        train_iterator, 
        eval_iterator, 
        policy_engine = policy_engine, 
        reference_engine = reference_engine, 
        reward_engine = reward_engine,
        critic_engine = critic_engine
    )

    log_message_rank0("#"*100+'\nTrainer has been built\n'+'#'*100, config.log_file, config.global_rank)
    return trainer

def setup_dsconfig(ds_config, config):
    #setup some deepspeed parameters from global_config
    ds_config['train_batch_size'] = config.global_batch_size
    ds_config['optimizer']['params']['lr']=config.learning_rate
    #The learning rate warmup from 0 to final leraning rate in 10% steps
    ds_config['scheduler']['params']['warmup_max_lr']=config.learning_rate
    ds_config['scheduler']['params']['warmup_min_lr']=0
    ds_config['scheduler']['params']['warmup_num_steps']=config.warmup_steps
    ds_config['gradient_accumulation_steps']=config.gradient_accumulation_steps
    ds_config['gradient_clipping']=config.max_grad_norm
    
    return ds_config

def load_models(config):
    """Load four models (if have) and initialize these models (if needed)"""
    reference_kwargs = policy_kwargs = {'torch_dtype' : getattr(torch, config.policy_dtype)}
    critic_kwargs = reward_kwargs = {'torch_dtype' : getattr(torch, config.reward_dtype)}
    log_message_rank0('building models...', config.log_file, config.global_rank)
    policy_model = reference_model = reward_model = critic_model = None
    policy_engine = reward_engine = reference_engine = critic_engine = None
    #This is the stage3 deepspeed config, it's the default deepspeed config for reference/reward/critic models
    #Since these models won't generate sentences during training, we can safely set the deepspeed zero optimization to stage-3
    stage3_config = json.load(open("configs/stage3_config.json"))
    stage3_config = setup_dsconfig(stage3_config, config)
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Rank of the LoRA update matrices
        lora_alpha=32,
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "lm_head"], # Modify based on your model architecture
        lora_dropout=0.1,
        bias="none"
    )
    scalarhead_lora_config = LoraConfig(
        r=8,  # Rank of the LoRA update matrices
        lora_alpha=32,
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "summary"], # Modify based on your model architecture
        lora_dropout=0.1,
        bias="none"
    )
    #SFT, DPO~, PPO will use the policy model
    if config.use_policy:
        #The config.model_path is the pretrained model path
        policy_model = AutoModelForCausalLM.from_pretrained(config.model_path, low_cpu_mem_usage=True, attn_implementation=config.flash_attention, **policy_kwargs, trust_remote_code=True)
        if config.use_lora:
            policy_model = get_peft_model(policy_model, lora_config)
        #Note that if we need the policy model to generate sentences we must set the zero stage of deepspeed to <=2
        #Because stage-3 will split the parameters of model, which will cause unacceptable network delay during generation procedure 
        policy_config = json.load(open("configs/stage2_config.json")) if (config.online or config.sample_ontest or config.loss_name=='ppo' or config.loss_name=='gen_refs') else json.load(open("configs/stage3_config.json"))
        policy_config = setup_dsconfig(policy_config, config)
        
        #initialize policy model if needed, in PPO/DPO ... the policy model is initialized as sft model, the policy path is the path to sft model
        #In sft training, the policy_path is None
        if config.policy_path is not None:
            log_message_rank0(f"{'#'*100}\nLoading from {config.policy_path} with tag {config.policy_tag}\n{'#'*100}", config.log_file, config.global_rank)
            state_dict = torch.load(os.path.join(config.policy_path, config.policy_tag+".pt"), weights_only=True)
            policy_model.load_state_dict(state_dict)
            policy_model = policy_model.to(policy_kwargs['torch_dtype'])
            del state_dict

        #build deepspeed model engine
        log_message_rank0("policy model deepspeed config is:\n"+json.dumps(policy_config, indent=4), config.log_file, config.global_rank)
        policy_engine, _, _, _ = deepspeed.initialize(model=policy_model, model_parameters=policy_model.parameters(), config=policy_config)
    
    #PPO, DPO~ will use the reference model
    if config.use_reference:
        reference_model = AutoModelForCausalLM.from_pretrained(config.model_path, low_cpu_mem_usage=True, attn_implementation=config.flash_attention, **reference_kwargs, trust_remote_code=True)
        if config.use_lora:
            reference_model = get_peft_model(reference_model, lora_config)
        #initialize reference model to sft model
        if config.reference_path is not None:
            log_message_rank0(f"{'#'*100}\nLoading from {config.reference_path} with tag {config.reference_tag}\n{'#'*100}", config.log_file, config.global_rank)
            state_dict = torch.load(os.path.join(config.reference_path, config.reference_tag+".pt"), weights_only=True)
            reference_model.load_state_dict(state_dict)
            reference_model = reference_model.to(reference_kwargs['torch_dtype'])
            del state_dict
    
        log_message_rank0("reference model deepspeed config is:\n"+json.dumps(stage3_config, indent=4), config.log_file, config.global_rank)
        reference_engine, _, _, _ = deepspeed.initialize(model=reference_model, model_parameters=reference_model.parameters(), config=stage3_config)
    
    #reward modeling, online Alignment will use the reward model, we never use lora to train reward model
    if config.use_reward:
        #Reward model need a scalar head to output a scalar
        if config.reward_odin:
            reward_model = AutoModelForCausalLMWithScalarHeadODIN.from_pretrained(config.model_path, low_cpu_mem_usage=True, attn_implementation=config.flash_attention, **reward_kwargs, trust_remote_code=True)
        else:
            reward_model = AutoModelForCausalLMWithScalarHead.from_pretrained(config.model_path, low_cpu_mem_usage=True, attn_implementation=config.flash_attention, **reward_kwargs, trust_remote_code=True)
        if config.use_lora:
            reward_model = get_peft_model(reward_model, scalarhead_lora_config)

        #In online Alignment, the reward model is initialized to 'already trained' reward model, and frozen its parameters during training
        #In reward training, the reward_path is None, that is we initialize reward model as pretrained model
        if config.reward_path is not None:
            log_message_rank0(f"{'#'*100}\nLoading from {config.reward_path} with tag {config.reward_tag}\n{'#'*100}", config.log_file, config.global_rank)
            state_dict = torch.load(os.path.join(config.reward_path, config.reward_tag+".pt"), weights_only=True)
            reward_model.load_state_dict(state_dict)
            reward_model = reward_model.to(reward_kwargs['torch_dtype'])
            del state_dict

        log_message_rank0("reward model deepspeed config is:\n"+json.dumps(stage3_config, indent=4), config.log_file, config.global_rank)  
        reward_engine, _, _, _ = deepspeed.initialize(model=reward_model, model_parameters=reward_model.parameters(), config=stage3_config)

    #only ppo will use the critic model, we never use lora to train critic model
    if config.use_critic:
        #Critic model need a scalar head to output a scalar
        if config.reward_odin:
            critic_model = AutoModelForCausalLMWithScalarHeadODIN.from_pretrained(config.model_path, low_cpu_mem_usage=True, attn_implementation=config.flash_attention, **critic_kwargs, trust_remote_code=True)
        else:
            critic_model = AutoModelForCausalLMWithScalarHead.from_pretrained(config.model_path, low_cpu_mem_usage=True, attn_implementation=config.flash_attention, **critic_kwargs, trust_remote_code=True)
        
        if config.use_lora:
            critic_model = get_peft_model(critic_model, scalarhead_lora_config)

        #critic model is initialized as 'already trained' reward model
        if config.critic_path is not None:
            log_message_rank0(f"{'#'*100}\nLoading from {config.critic_path} with tag {config.critic_tag}\n{'#'*100}", config.log_file, config.global_rank)
            state_dict = torch.load(os.path.join(config.critic_path, config.critic_tag+".pt"), weights_only=True)
            critic_model.load_state_dict(state_dict)
            critic_model = critic_model.to(critic_kwargs['torch_dtype'])
            del state_dict

        #critic model will use a different learning rate from policy model   
        stage3_config['optimizer']['params']['lr']=config.critic_lr
        stage3_config['scheduler']['params']['warmup_max_lr']=config.critic_lr
        log_message_rank0("critic model deepspeed config is:\n"+json.dumps(stage3_config, indent=4), config.log_file, config.global_rank)
        critic_engine, _, _, _ = deepspeed.initialize(model=critic_model, model_parameters=critic_model.parameters(), config=stage3_config)
    

    tokenizer_path = config.tokenizer_path or config.model_path
    log_message_rank0(f'Loading tokenizer {tokenizer_path}', config.log_file, config.global_rank)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer, policy_engine, reward_engine, reference_engine, critic_engine

def setup_global_config():
    """setup some global config parameters before training"""
    config = get_args()
    set_seed(config.seed)
    #default running directory name will be set by loss_name, model_name, dataset_name
    if config.exp_name == None:
        config.exp_name = config.loss_name + "_" + config.model_name + "_" + config.dataset
    if config.local_run_dir == None:
        config.local_run_dir = os.path.join('/data/models', config.exp_name)
    if config.remote_run_dir == None:
        config.remote_run_dir = config.local_run_dir
    
    if config.n_examples!=None:
        dataset_len = config.n_examples
    elif config.loss_name == 'sft':
        dataset_len = getattr(dataset, f'get_{config.dataset}_len')('train_sft')*config.n_epochs
    else:
        dataset_len = getattr(dataset, f'get_{config.dataset}_len')('train_prefs')*config.n_epochs


    config.num_prompts = dataset_len
    config.local_rank = int(os.getenv('LOCAL_RANK'))
    config.global_rank = int(os.getenv('RANK'))
    config.world_size = int(os.getenv('WORLD_SIZE'))
    #warmup during 0.1 epoch at the begining
    config.warmup_steps = int(0.1*(config.num_prompts/config.global_batch_size))
    #evaluate model on test set every 1/10 epoch
    config.eval_every = int(0.1*config.num_prompts)
    config.sample_every = 1
    #save checkpoint every 1/10 epoch for online, every 1/5 epoch for offline, if --intermediate_checkpoints is on
    if config.online or config.loss_name=='ppo':
        config.save_every = int(0.1*config.num_prompts)
    else:
        #PPO and offline
        config.save_every = int(0.2*config.num_prompts)

    #online DPO~ will have 'PromptDataLoader' as train/test DataLoader
    if config.online:
        config.train_dataloader = 'PromptDataLoader'
        config.test_dataloader = 'PromptDataLoader'
        config.use_reward = True

    #logfile only on one process
    if config.global_rank == 0:
        config.log_file = setup_logger()

    #remake directories
    if os.path.exists(config.local_run_dir) and config.global_rank==0 and not config.eval_only  and not config.reward_statistics:
        os.system(f'rm -rf {config.local_run_dir}')
    if os.path.exists(config.remote_run_dir) and config.global_rank==0 and not config.eval_only and not config.reward_statistics:
        os.system(f'rm -rf {config.remote_run_dir}')

    if config.global_rank==0:
        os.makedirs(config.local_run_dir, exist_ok=True)
        os.makedirs(config.remote_run_dir, exist_ok=True)

    log_message_rank0(f"Making experiment directory {config.local_run_dir} and {config.remote_run_dir}", config.log_file, config.global_rank)

    #Ensure that config.eval_every can be divided by config.global_batch_size without a remainder.
    if config.eval_every % config.global_batch_size != 0:
        log_message_rank0(f'Setting eval_every to {config.eval_every - config.eval_every % config.global_batch_size}', config.log_file, config.global_rank)
        config.eval_every = config.eval_every - config.eval_every % config.global_batch_size

    #Ensure that config.save_every can be divided by config.eval_every without a remainder.
    if config.save_every % config.eval_every != 0:
        log_message_rank0(f'Setting save_every to {config.save_every - config.save_every % config.eval_every}', config.log_file, config.global_rank)
        config.save_every = config.save_every - config.save_every % config.eval_every
    
    config_dict = vars(config)  # convert Namespace to dict
    log_message_rank0(json.dumps(config_dict, indent=4), config.log_file, config.global_rank)
    return config

def network_test(config):
    """test the network between processes/GPUs"""
    num_elements = (1024**3) // 4
    for i in range(10):
        # Create a tensor with the size of num_elements, filled with 1
        tensor = torch.ones(num_elements, dtype=torch.float32).to(config.local_rank)
        all_gather_if_needed(tensor, config.local_rank, config.world_size)
        log_message_rank0(f"iter-{i+1}, 1GB tensor gather successful!", config.log_file, config.global_rank)
    
    del tensor

def main():
    #setup config
    config = setup_global_config()
    setup_distributed(config.local_rank)
    if config.network_test:
        network_test(config)
    # load models and tokenizer
    tokenizer, policy_engine, reward_engine, reference_engine, critic_engine  = load_models(config)
    # build up the trainer
    trainer = build_trainer(config, tokenizer, policy_engine, reward_engine, reference_engine, critic_engine)
   
    #only do evaluation on test set or train the model on training set?
    if config.eval_only:
        trainer.eval_ontest(config.policy_tag)
        if config.sample_ontest:
            trainer.sample_ontest(config.policy_tag)
    else:
        trainer.train()

    cleanup_distributed()

if __name__ == '__main__':
    main()
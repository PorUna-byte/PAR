import argparse
import os

############################################################
def setup_basic_argparse(parser):
    # Random seed
    parser.add_argument('--seed', type=int, default=22, help='Random seed for initialization')

    # Experiment name
    parser.add_argument('--exp_name', type=str, default=None, help='Name for this experiment')

    # Datasets
    parser.add_argument('--dataset', type=str, required=True, help='Datasets to be used for training')

    # Wandb configuration
    parser.add_argument('--wandb_enabled', action='store_true', help='Enable logging to Weights & Biases')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity')
    parser.add_argument('--wandb_project', type=str, default="rh-research-online", help='Weights & Biases project name')
    parser.add_argument('--cache_dir', type=str, default='/data/models', help='Directory where wandb cache will be saved')

    # Local run directory
    parser.add_argument('--local_run_dir', type=str,  default=None, help='Directory for saving sample_on_test, set at runtime based on exp_name')
    # Remote run directory
    parser.add_argument('--remote_run_dir', type=str, default=None, help='Directory for saving model checkpoints, set a runtime based on exp_name')

    # Initial evaluation
    parser.add_argument('--no_first_eval', action='store_true', help='Whether to evaluate at the very beginning of training')

    # Minimum log interval for wandb
    parser.add_argument('--minimum_log_interval_secs', type=float, default=10.0, help='Minimum seconds between logging events to wandb')

    # Intermediate checkpoints
    parser.add_argument('--save_ckps', type=str, default="", help='When to save checkpoints')
    parser.add_argument('--save_ckps_val', type=list, default=[], help='Will be set automatically')
    parser.add_argument('--no_save_latest', action='store_true', help='Wheather to save latest checkpoint')

    # Training settings, all these parameters will be set automatically by code
    parser.add_argument('--warmup_steps', type=int, default=None, help='Number of linear warmup steps for the learning rate')
    parser.add_argument('--eval_every', type=int, default=None, help='Evaluate model every <eval_every> steps')
    parser.add_argument('--num_prompts', type=int, default=None, help='The number of prompts for one epoch, Should be changed according to your datasets')

    # Evaluation settings
    parser.add_argument('--n_eval_examples', type=int, default=None, help='Number of examples to evaluate on (leave as null to evaluate on all of them)')

    # Sampling settings
    parser.add_argument('--top_p', type=float, default=0.9, help='Top p choice for sampling')
    parser.add_argument('--temperature', type=float, default=0.9, help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=50, help='Top k choices for sampling')
    parser.add_argument('--length_penalty', type=float, default=2, help='The length penalty for sampling')

    # DataLoader settings
    parser.add_argument('--human_prefix', type=str, default="<|user|>", help='Label to use for human turns in chat datasets')
    parser.add_argument('--assistant_prefix', type=str, default="<|assistant|>", help='Label to use for assistant turns in chat datasets')
    parser.add_argument('--human_suffix', type=str, default="", help='Used to mark end of human turn')
    parser.add_argument('--assistant_suffix', type=str, default="", help='Used to mark end of assistant turn')

    # model name and loss name
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model, shold be llama3-8b or gemma2-2b')
    parser.add_argument('--loss_name', type=str, required=True, help='Name of the loss, should be ppo,dpo,slic or kto')
    parser.add_argument('--log_file', type=str, default=None, help="the path of log file, will be set by code")

    # flags
    parser.add_argument('--network_test', type=bool, default=True, help='Whether to test network beforehand')
    parser.add_argument('--reward_penalize_length', type=bool, default=True, help='Whether to penalize the lengthy response')

    parser.add_argument('--online', action='store_true', help='Whether to use online training')
    parser.add_argument('--eval_only', action='store_true', help='Whether to evalute only on test set, no training')
    parser.add_argument('--sample_ontest', action='store_true', help='Whether to sample on test set when do offline training')
    parser.add_argument('--reward_statistics', action='store_true', help='Train the reward model or use already trained model to calculate reward statistics on training set')
    
    #KL penalty
    parser.add_argument('--KL_coef', type=float, default=0.005, help='Coefficient on KL penalty')
    #dpo specific settings
    parser.add_argument('--dpo_beta', type=float, default=0.1, help='The beta for DPO training')
    #ipo specific settings
    parser.add_argument('--ipo_tau', type=float, default=0.1, help='The tau for IPO training')

    #Normalize the reward
    parser.add_argument('--reward_reg', action='store_true', help='Whether to use regularization term for reward training')
    parser.add_argument('--reward_reg_val', type=float, default=0.005, help='The degree of regularization term')
    parser.add_argument('--reward_ceil', type=float, default=None, help='The Ceil of reward')

    parser.add_argument('--reward_odin', action='store_true', help='Whether to train the reward model via ODIN')
    parser.add_argument('--reward_meanstd', action='store_true', help='Whether to reshape the reward via the running mean and std')
    parser.add_argument('--reward_minmax', action='store_true', help='Whether to reshape the reward via the running max and min')

    parser.add_argument('--reward_relative', action='store_true', help='Whether to reshape the reward via the reference rewards')
    parser.add_argument('--reward_tanh', action='store_true', help='Whether to reshape the reward via the tanh')
    parser.add_argument('--reward_fittedpoly', action='store_true', help='Whether to reshape the reward via the fitted polynomial function')
    parser.add_argument('--reward_maxref', type=int, default=10, help='How many reference rewards are needed to calculate the relative winrate')

    parser.add_argument('--reward_sigmoid', action='store_true', help='Whether to reshape the reward via sigmoid fuction')
    parser.add_argument('--reward_centered', action='store_true', help='Whether to reshape the reward via reference rewards(vanilla centered)')
    parser.add_argument('--reward_sgfc', action='store_true', help='Whether to reshape the reward via slow growth then fast converge')

    parser.add_argument('--sigmoid_k', type=int, default=1, help='The parameter k of sigmoid function')
    parser.add_argument('--reward_lsc', action='store_true', help='Whether to reshape the reward via log-sigmoid centered')
    parser.add_argument('--reward_clipping', action='store_true', help='Whether to clip the reward using pre-calculated mean and std')
    return parser

#################################################
def setup_model_argparse(parser):
    # Model and tokenizer settings
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Name or path of the tokenizer to use; if null, will use the tokenizer from the model')

    # Override pre-trained weights
    parser.add_argument('--policy_path', type=str, default=None,
                        help='Override pre-trained weights of policy model')
    parser.add_argument('--policy_tag', type=str, default='latest_hf',
                        help='Override pre-trained weights of policy model at a specific checkpoint')
    parser.add_argument('--reward_path', type=str, default=None,
                        help='Override pre-trained weights of reward model')
    parser.add_argument('--reward_tag', type=str, default='latest_hf',
                        help='Override pre-trained weights of reward model at a specific checkpoint')
    parser.add_argument('--reference_path', type=str, default=None,
                        help='Override pre-trained weights of reference model')
    parser.add_argument('--reference_tag', type=str, default='latest_hf',
                        help='Override pre-trained weights of reference model at a specific checkpoint')
    parser.add_argument('--critic_path', type=str, default=None,
                        help='Override pre-trained weights of critic model')
    parser.add_argument('--critic_tag', type=str, default='latest_hf',
                        help='Override pre-trained weights of critic model at a specific checkpoint')
    
    # Module and precision settings
    parser.add_argument('--policy_dtype', type=str, default='bfloat16',
                        help='Data type for the policy/reference parameters/optimizer state')
    parser.add_argument('--reward_dtype', type=str, default='bfloat16',
                        help='Data type for the reward/critic model')

    # Training specifics
    parser.add_argument('--global_batch_size', type=int, default=64,
                        help='The total batch_size for all GPUs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate over for each batch')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for policy model')
    parser.add_argument('--critic_lr', type=float, default=1e-5, help='Learning rate for critic model')
    parser.add_argument('--max_grad_norm', type=float, default=10.0,
                        help='Maximum gradient norm to clip to')
    parser.add_argument('--use_lora', type=bool, default=False, help='Whether to use lora')
    
    parser.add_argument('--n_epochs', type=int, default=1, help='The number of epochs to train for; if null, must specify n_examples')
    parser.add_argument('--n_examples', type=int, default=None, help='The number of examples to train for; if null, must specify n_epochs')
    return parser

def setup_hh_rlhf_argparse(parser):
    # reward normalization
    parser.add_argument('--gen_valid_len', type=int, default=300, help='the threshold of length to be penalized')
    parser.add_argument('--len_penalty', type=float, default=0.01, help='the penalty of each token that exceed the gen_valid_len')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum allowed length for an input (prompt + response)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum allowed generated tokens for an prompt')
    parser.add_argument('--max_prompt_length', type=int, default=512,
                        help='Maximum allowed length for a prompt (remainder will be dedicated to the completion)')
    return parser
    

def setup_ultrafb_bin_argparse(parser):
    # reward normalization 
    parser.add_argument('--gen_valid_len', type=int, default=300, help='the threshold of length to be penalized')
    parser.add_argument('--len_penalty', type=float, default=0.01, help='the penalty of each token that exceed the gen_valid_len')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum allowed length for an input (prompt + response)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum allowed generated tokens for an prompt')
    parser.add_argument('--max_prompt_length', type=int, default=512,
                        help='Maximum allowed length for a prompt (remainder will be dedicated to the completion)')
    return parser
    
##########################################################
def setup_llama3_8b_argparse(parser):
    parser.add_argument('--model_path', type=str, default="/data/models/Llama-3-8B",
                        help='Name or path of the pretrained model to use')
    parser.add_argument('--flash_attention', type=str, default="flash_attention_2", help='Which flash attention to use')
    return parser

def setup_gemma2_2b_argparse(parser):
    parser.add_argument('--model_path', type=str, default="/data/models/Gemma-2-2B",
                        help='Name or path of the pretrained model to use')
    parser.add_argument('--flash_attention', type=str, default="eager", help='Which flash attention to use')
    return parser

#############################################################
def setup_ppo_argparse(parser):
    # PPO specific settings
    parser.add_argument('--buffer_size', type=int, default=4, help='Number of batchs in the replay buffer')
    parser.add_argument('--cliprange', type=float, default=0.2, help='Used to clip the probability ratio in range [1-cliprange, 1+cliprange]')
    parser.add_argument('--lam', type=float, default=0.95, help='Lambda for PPO')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma for PPO')
    parser.add_argument('--critic_eps', type=float, default=0.2, help='clip range on critic loss in PPO')

    # Trainer and DataLoader settings
    parser.add_argument('--trainer', type=str, default='PPOTrainer', help='The trainer class to use')
    parser.add_argument('--train_dataloader', type=str, default='PromptDataLoader', help='The DataLoader class to use')
    parser.add_argument('--test_dataloader', type=str, default='PromptDataLoader', help='The DataLoader class to use')

    # use models
    parser.add_argument('--use_policy', type=bool, default=True, help='whether to use policy model')
    parser.add_argument('--use_reference', type=bool, default=True, help='whether to use reference model')
    parser.add_argument('--use_reward', type=bool, default=True, help='whether to use reward model')
    parser.add_argument('--use_critic', type=bool, default=True, help='whether to use critic model')

    return parser

def setup_grpo_argparse(parser):
    # GRPO specific settings
    parser.add_argument('--buffer_size', type=int, default=4, help='Number of batchs in the replay buffer')
    parser.add_argument('--group_size', type=int, default=5, help='Number of answers in the same group(i.e. for the same question)')
    parser.add_argument('--cliprange', type=float, default=0.2, help='Used to clip the probability ratio in range [1-cliprange, 1+cliprange]')

    # Trainer and DataLoader settings
    parser.add_argument('--trainer', type=str, default='GRPOTrainer', help='The trainer class to use')
    parser.add_argument('--train_dataloader', type=str, default='PromptDataLoader', help='The DataLoader class to use')
    parser.add_argument('--test_dataloader', type=str, default='PromptDataLoader', help='The DataLoader class to use')

    # use models
    parser.add_argument('--use_policy', type=bool, default=True, help='whether to use policy model')
    parser.add_argument('--use_reference', type=bool, default=True, help='whether to use reference model')
    parser.add_argument('--use_reward', type=bool, default=True, help='whether to use reward model')
    parser.add_argument('--use_critic', type=bool, default=False, help='whether to use critic model')

    return parser

def setup_remax_argparse(parser):
    # Trainer and DataLoader settings
    parser.add_argument('--trainer', type=str, default='ReMaxTrainer', help='The trainer class to use')
    parser.add_argument('--train_dataloader', type=str, default='PromptDataLoader', help='The DataLoader class to use')
    parser.add_argument('--test_dataloader', type=str, default='PromptDataLoader', help='The DataLoader class to use')

    # use models
    parser.add_argument('--use_policy', type=bool, default=True, help='whether to use policy model')
    parser.add_argument('--use_reference', type=bool, default=False, help='whether to use reference model')
    parser.add_argument('--use_reward', type=bool, default=True, help='whether to use reward model')
    parser.add_argument('--use_critic', type=bool, default=False, help='whether to use critic model')

    return parser

def setup_dpo_argparse(parser):
    parser.add_argument('--avg_logp', type=bool, default=False, help='Whether to average logp when calculating logp on models')

    # Trainer and DataLoader settings
    parser.add_argument('--trainer', type=str, default='DPOTrainer', help='The trainer class to use')
    parser.add_argument('--train_dataloader', type=str, default='PairedPreferenceDataLoader', help='The DataLoader class to use')
    parser.add_argument('--test_dataloader', type=str, default='PairedPreferenceDataLoader', help='The DataLoader class to use')
    
    # use models
    parser.add_argument('--use_policy', type=bool, default=True, help='whether to use policy model')
    parser.add_argument('--use_reference', type=bool, default=True, help='whether to use reference model')
    parser.add_argument('--use_reward', type=bool, default=True, help='whether to use reward model, default to offline dpo')
    parser.add_argument('--use_critic', type=bool, default=False, help='whether to use critic model')
    return parser

def setup_ipo_argparse(parser):
    #ipo specific settings
    parser.add_argument('--avg_logp', type=bool, default=True, help='Whether to average logp when calculating logp on models')

    # Trainer and DataLoader settings
    parser.add_argument('--trainer', type=str, default='IPOTrainer', help='The trainer class to use')
    parser.add_argument('--train_dataloader', type=str, default='PairedPreferenceDataLoader', help='The DataLoader class to use')
    parser.add_argument('--test_dataloader', type=str, default='PairedPreferenceDataLoader', help='The DataLoader class to use')
    
    # use models
    parser.add_argument('--use_policy', type=bool, default=True, help='whether to use policy model')
    parser.add_argument('--use_reference', type=bool, default=True, help='whether to use reference model')
    parser.add_argument('--use_reward', type=bool, default=True, help='whether to use reward model, default to offline ipo')
    parser.add_argument('--use_critic', type=bool, default=False, help='whether to use critic model')
    return parser

def setup_kto_argparse(parser):
    #kto specific settings
    parser.add_argument('--beta', type=float, default=0.01, help='The beta for KTO training')
    parser.add_argument('--avg_logp', type=bool, default=False, help='Whether to average logp when calculating logp on models')

    # Trainer and DataLoader settings
    parser.add_argument('--trainer', type=str, default='KTOTrainer', help='The trainer class to use')
    parser.add_argument('--train_dataloader', type=str, default='UnpairedPreferenceDataLoader', help='The DataLoader class to use')
    parser.add_argument('--test_dataloader', type=str, default='UnpairedPreferenceDataLoader', help='The DataLoader class to use')
    
    # use models
    parser.add_argument('--use_policy', type=bool, default=True, help='whether to use policy model')
    parser.add_argument('--use_reference', type=bool, default=True, help='whether to use reference model')
    parser.add_argument('--use_reward', type=bool, default=True, help='whether to use reward model, default to offline kto')
    parser.add_argument('--use_critic', type=bool, default=False, help='whether to use critic model')
    return parser

def setup_slic_argparse(parser):
    #slic specific settings
    parser.add_argument('--beta', type=float, default=1.0, help='The beta for SLIC training')
    parser.add_argument('--lam', type=float, default=0.1, help='The lambda for SLIC training')
    parser.add_argument('--avg_logp', type=bool, default=True, help='Whether to average logp when calculating logp on models')

    # Trainer and DataLoader settings
    parser.add_argument('--trainer', type=str, default='SLiCTrainer', help='The trainer class to use')
    parser.add_argument('--train_dataloader', type=str, default='PairedPreferenceDataLoader', help='The DataLoader class to use')
    parser.add_argument('--test_dataloader', type=str, default='PairedPreferenceDataLoader', help='The DataLoader class to use')
    
    # use models
    parser.add_argument('--use_policy', type=bool, default=True, help='whether to use policy model')
    parser.add_argument('--use_reference', type=bool, default=False, help='whether to use reference model')
    parser.add_argument('--use_reward', type=bool, default=True, help='whether to use reward model, default to offline dpo')
    parser.add_argument('--use_critic', type=bool, default=False, help='whether to use critic model')
    return parser

def setup_reward_argparse(parser):
    parser.add_argument('--avg_logp', type=bool, default=False, help='Whether to average logp when calculating logp on models')
    # Trainer and DataLoader settings
    parser.add_argument('--trainer', type=str, default='RewardTrainer', help='The trainer class to use')
    parser.add_argument('--train_dataloader', type=str, default='PairedPreferenceDataLoader', help='The DataLoader class to use')
    parser.add_argument('--test_dataloader', type=str, default='PairedPreferenceDataLoader', help='The DataLoader class to use')
    # use models
    parser.add_argument('--use_policy', type=bool, default=False, help='whether to use policy model')
    parser.add_argument('--use_reference', type=bool, default=False, help='whether to use reference model')
    parser.add_argument('--use_reward', type=bool, default=True, help='whether to use reward model')
    parser.add_argument('--use_critic', type=bool, default=False, help='whether to use critic model')
    return parser

def setup_reward_odin_argparse(parser):
    parser.add_argument('--avg_logp', type=bool, default=False, help='Whether to average logp when calculating logp on models')
    parser.add_argument('--reward_odin_L', type=float, default=1.0, help='The weight of length loss')
    parser.add_argument('--reward_odin_O', type=float, default=1.0, help='The weight of orthogonal loss')
    # Trainer and DataLoader settings
    parser.add_argument('--trainer', type=str, default='RewardTrainerODIN', help='The trainer class to use')
    parser.add_argument('--train_dataloader', type=str, default='PairedPreferenceDataLoader', help='The DataLoader class to use')
    parser.add_argument('--test_dataloader', type=str, default='PairedPreferenceDataLoader', help='The DataLoader class to use')
    # use models
    parser.add_argument('--use_policy', type=bool, default=False, help='whether to use policy model')
    parser.add_argument('--use_reference', type=bool, default=False, help='whether to use reference model')
    parser.add_argument('--use_reward', type=bool, default=True, help='whether to use reward model')
    parser.add_argument('--use_critic', type=bool, default=False, help='whether to use critic model')
    return parser

def setup_sft_argparse(parser):
    parser.add_argument('--avg_logp', type=bool, default=True, help='Whether to average logp when calculating logp on models')
    # Trainer and DataLoader settings
    parser.add_argument('--trainer', type=str, default='SFTTrainer', help='The trainer class to use')
    parser.add_argument('--train_dataloader', type=str, default='SFTDataLoader', help='The Training DataLoader class to use')
    parser.add_argument('--test_dataloader', type=str, default='SFTDataLoader', help='The Test DataLoader class to use')
    
    # use models
    parser.add_argument('--use_policy', type=bool, default=True, help='whether to use policy model')
    parser.add_argument('--use_reference', type=bool, default=False, help='whether to use reference model')
    parser.add_argument('--use_reward', type=bool, default=False, help='whether to use reward model')
    parser.add_argument('--use_critic', type=bool, default=False, help='whether to use critic model')

    return parser

def setup_gen_refs_argparse(parser):
    parser.add_argument('--avg_logp', type=bool, default=True, help='Whether to average logp when calculating logp on models')
    parser.add_argument('--num_refs', type=int, default=10, help='How many reference responses should be generated for each prompt?')
    # Trainer and DataLoader settings
    parser.add_argument('--trainer', type=str, default='GenrefsTrainer', help='The trainer class to use')
    parser.add_argument('--train_dataloader', type=str, default='PromptDataLoader', help='The Training DataLoader class to use')
    parser.add_argument('--test_dataloader', type=str, default='PromptDataLoader', help='The Test DataLoader class to use')
    
    # use models
    parser.add_argument('--use_policy', type=bool, default=True, help='whether to use policy model')
    parser.add_argument('--use_reference', type=bool, default=False, help='whether to use reference model')
    parser.add_argument('--use_reward', type=bool, default=True, help='whether to use reward model')
    parser.add_argument('--use_critic', type=bool, default=False, help='whether to use critic model')
    return parser

def setup_preference_argparse(parser):
    parser.add_argument('--avg_logp', type=bool, default=False, help='Whether to average logp when calculating logp on models')
    # Trainer and DataLoader settings
    parser.add_argument('--trainer', type=str, default='PreferenceTrainer', help='The trainer class to use')
    parser.add_argument('--train_dataloader', type=str, default='SFTDataLoader', help='The Training DataLoader class to use')
    parser.add_argument('--test_dataloader', type=str, default='SFTDataLoader', help='The Test DataLoader class to use')
    
    # use models
    parser.add_argument('--use_policy', type=bool, default=True, help='whether to use policy model')
    parser.add_argument('--use_reference', type=bool, default=False, help='whether to use reference model')
    parser.add_argument('--use_reward', type=bool, default=False, help='whether to use reward model')
    parser.add_argument('--use_critic', type=bool, default=False, help='whether to use critic model')
    return parser
#############################################

def setup_distributed_argparse():
    #prepare deepspeed parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    parser.add_argument('--global_rank', type=int, default=-1,
                    help='global rank set by environment variable')
    parser.add_argument('--world_size', type=int, default=-1,
                         help='global world_size set by environment variable')

    return parser

# Use this function to set up argparse
def get_args():
    parser = setup_distributed_argparse()
    parser = setup_basic_argparse(parser)
    parser = setup_model_argparse(parser)
    args = parser.parse_args() 

    if args.model_name=='llama3-8b':
        parser = setup_llama3_8b_argparse(parser)
    elif args.model_name=='gemma2-2b':
        parser = setup_gemma2_2b_argparse(parser)
    else:
        raise Exception(f"Unsupported model {args.model_name}")
    
    if args.dataset=='hh_rlhf':
        parser = setup_hh_rlhf_argparse(parser)
    elif args.dataset=='ultrafb_bin':
        parser = setup_ultrafb_bin_argparse(parser)
    else:
        raise Exception(f"Unsupported dataset {args.dataset}")

    if args.loss_name=='sft':
        parser = setup_sft_argparse(parser)
    elif args.loss_name=='preference':
        parser = setup_preference_argparse(parser)
    elif args.loss_name=='reward':
        parser = setup_reward_argparse(parser)
    elif args.loss_name=='reward_odin':
        parser = setup_reward_odin_argparse(parser)
    elif args.loss_name=='ppo':
        parser = setup_ppo_argparse(parser)
    elif args.loss_name=='grpo':
        parser = setup_grpo_argparse(parser)
    elif args.loss_name=='remax':
        parser = setup_remax_argparse(parser)
    elif args.loss_name=='dpo':
        parser = setup_dpo_argparse(parser)
    elif args.loss_name=='kto':
        parser = setup_kto_argparse(parser)
    elif args.loss_name=='ipo':
        parser = setup_ipo_argparse(parser)
    elif args.loss_name=='slic':
        parser = setup_slic_argparse(parser)
    elif args.loss_name=='gen_refs':
        parser = setup_gen_refs_argparse(parser)
    else:
        raise Exception(f"Unsupported loss {args.loss_name}")

    args = parser.parse_args()
    return args
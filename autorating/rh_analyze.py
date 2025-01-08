from datamerge import DataMerge
from llmscorer import LLMScorer
from figuredraw import FigureDraw
import argparse


def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_exp_name', type=str, default="['ppo_gemma2-2b_ultrafb_bin_vanilla']", help='The experiment name for policy model samples on test')
    parser.add_argument('--sft_exp_name', type=str, default='sft_gemma2-2b_ultrafb_bin', help='The experiment name for sft model samples on test')
    parser.add_argument('--base_model_name', type=str, default='gemma2-2b', help='The base model for Alignment')
    parser.add_argument('--llm_model_name', type=str, default='deepseek-chat', help='The preference model for scoring')
    parser.add_argument('--temperature', type=float, default=0.1, help='The temperature for LLM scoring, set to 0.1 to ensure reproducibility')
    parser.add_argument('--generation_times', type=int, default=1, help='The repetition times for each scoring')
    parser.add_argument('--dataset_name', type=str, default='ultrafb_bin', help='The dataset we are using')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--merge_sft', action='store_true', help='whether to merge for sft')
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--human_prefix', type=str, default="<|user|>", help='Label to use for human turns in chat datasets')
    parser.add_argument('--assistant_prefix', type=str, default="<|assistant|>", help='Label to use for assistant turns in chat datasets')
    parser.add_argument('--human_suffix', type=str, default="", help='Used to mark end of human turn')
    parser.add_argument('--assistant_suffix', type=str, default="", help='Used to mark end of assistant turn')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = setup_argparse()
    ###########################   
    args.base_model_name = 'gemma2-2b'
    args.dataset_name = 'ultrafb_bin'
    # ['ppo_gemma2-2b_ultrafb_bin_vanilla', 'ppo_gemma2-2b_ultrafb_bin_sigmoid','ppo_gemma2-2b_ultrafb_bin_reg','ppo_gemma2-2b_ultrafb_bin_meanstd','ppo_gemma2-2b_ultrafb_bin_clip','ppo_gemma2-2b_ultrafb_bin_lsc', 'ppo_gemma2-2b_ultrafb_bin_par', 'ppo_gemma2-2b_ultrafb_bin_minmax']
    # ['vanilla', 'reg','WARM', 'ODIN', 'meanstd','clip','minmax','lsc','sigmoid','par']
    policy_exp_names = ['WARM', 'ODIN']
    policy_exp_names_full = []
    for policy_exp_name in policy_exp_names:
        args.policy_exp_name = 'ppo_' + args.base_model_name + "_" + args.dataset_name + "_" + policy_exp_name
        policy_exp_names_full.append(args.policy_exp_name)
        args.sft_exp_name = 'sft_' + args.base_model_name + '_' + args.dataset_name
        data_merger = DataMerge(policy_exp_name=args.policy_exp_name, sft_exp_name=args.sft_exp_name, merge_sft=True, eos_str='<eos>')
        data_merger.run()
        llmscorer = LLMScorer(args)
        llmscorer.run_policy_sft_score()

    args.policy_exp_name = str(policy_exp_names_full)
    drawer = FigureDraw(args)
    drawer.draw_rh_bystep()

    ###########################

    # args.dataset_name = 'hh_rlhf'
    # # ['ppo_gemma2-2b_hh_rlhf_vanilla', 'ppo_gemma2-2b_hh_rlhf_sigmoid','ppo_gemma2-2b_hh_rlhf_reg', 'ppo_gemma2-2b_hh_rlhf_stdmean']
    # policy_exp_names = ['ppo_gemma2-2b_hh_rlhf_vanilla', 'ppo_gemma2-2b_hh_rlhf_sigmoid']
    # for policy_exp_name in policy_exp_names:
    #     args.policy_exp_name = policy_exp_name
    #     args.sft_exp_name = 'sft_gemma2-2b_hh_rlhf'
    #     args.dataset_name = 'hh_rlhf'
    #     data_merger = DataMerge(policy_exp_name=args.policy_exp_name, sft_exp_name=args.sft_exp_name, merge_sft=True, eos_str='<eos>')
    #     data_merger.run()
    #     llmscorer = LLMScorer(args)
    #     llmscorer.run_policy_sft_score()

    # args.policy_exp_name = str(policy_exp_names)
    # drawer = FigureDraw(args)
    # drawer.draw_rh_bystep()



       








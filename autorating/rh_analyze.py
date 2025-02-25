from datamerge import DataMerge
from llmscorer import LLMScorer
from figuredraw import FigureDraw
import argparse
import concurrent.futures
import copy
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_exp_name', type=str, default="['ppo_gemma2-2b_ultrafb_bin_vanilla']", help='The experiment name for policy model samples on test')
    parser.add_argument('--display_name', type=str, default="['vanilla']", help='The display name on figures')
    parser.add_argument('--sft_exp_name', type=str, default='sft_gemma2-2b_ultrafb_bin', help='The experiment name for sft model samples on test')
    parser.add_argument('--eos_str', type=str, default='<eos>', help='The end of string token')
    parser.add_argument('--loss_name', type=str, default='ppo', help='The name of loss function')
    parser.add_argument('--base_model_name', type=str, default='gemma2-2b', help='The base model for Alignment')
    parser.add_argument('--llm_model_name', type=str, default='deepseek-chat', help='The preference model for scoring')
    parser.add_argument('--temperature', type=float, default=0.1, help='The temperature for LLM scoring, set to 0.1 to ensure reproducibility')
    parser.add_argument('--generation_times', type=int, default=1, help='The repetition times for each scoring')
    parser.add_argument('--dataset_name', type=str, default='ultrafb_bin', help='The dataset we are using')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--merge_sft', action='store_true', help='whether to merge for sft')
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--human_prefix', type=str, default="<|user|>", help='Label to use for human turns in chat datasets')
    parser.add_argument('--assistant_prefix', type=str, default="<|assistant|>", help='Label to use for assistant turns in chat datasets')
    parser.add_argument('--human_suffix', type=str, default="", help='Used to mark end of human turn')
    parser.add_argument('--assistant_suffix', type=str, default="", help='Used to mark end of assistant turn')
    parser.add_argument('--lab_name', type=str, default="", help='The name of experiment')
    args = parser.parse_args()
    return args

def process_policy_exp(args, policy_exp_name):
    policy_exp_name_full = args.loss_name + '_' + args.base_model_name + "_" + args.dataset_name + "_" + policy_exp_name
    llmscorer = LLMScorer(args, policy_exp_name_full)
    llmscorer.run_policy_sft_score()

def gemma_ultra_lab1():
    args = setup_argparse()
    args.base_model_name = 'gemma2-2b'
    args.eos_str = '<eos>'
    args.dataset_name = 'ultrafb_bin'
    args.loss_name = 'ppo'
    args.lab_name = 'lab1'
    policy_exp_names = ['vanilla', 'WARM', 'ODIN', 'reg','meanstd', 'clip', 'minmax', 'lsc', 'PAR','vanilla_centered', 'sgfc']
    args.display_name = str(['Vanilla', 'WARM', 'ODIN', 'Reg','Meanstd', 'Clip', 'Minmax', 'LSC', 'PAR','Vanilla_Centered', 'SgFc'])

    policy_exp_names_full = []
    #Merge
    for policy_exp_name in policy_exp_names:
        policy_exp_name_full = args.loss_name + '_' + args.base_model_name + "_" + args.dataset_name + "_" + policy_exp_name
        sft_exp_name = 'sft_' + args.base_model_name + '_' + args.dataset_name
        policy_exp_names_full.append(policy_exp_name_full) 
        data_merger = DataMerge(policy_exp_name=policy_exp_name_full, sft_exp_name=sft_exp_name, merge_sft=True, eos_str=args.eos_str)
        data_merger.run()

    #Score
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_policy_exp, args, policy_exp_name) for policy_exp_name in policy_exp_names]
        
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 获取任务结果（如果有）

    #draw
    args.policy_exp_name = str(policy_exp_names_full)
    drawer = FigureDraw(args)
    drawer.draw_rh_bystep()

def gemma_ultra_lab1_grpo():
    args = setup_argparse()
    args.base_model_name = 'gemma2-2b'
    args.eos_str = '<eos>'
    args.dataset_name = 'ultrafb_bin'
    args.loss_name = 'grpo'
    args.lab_name = 'lab1'
    policy_exp_names = ['vanilla', 'WARM', 'ODIN', 'reg','meanstd', 'clip', 'minmax', 'lsc', 'PAR']
    args.display_name = str(['Vanilla', 'WARM', 'ODIN', 'Reg','Meanstd', 'Clip', 'Minmax', 'LSC', 'PAR'])
    policy_exp_names_full = []
    #Merge
    for policy_exp_name in policy_exp_names:
        policy_exp_name_full = args.loss_name + '_' + args.base_model_name + "_" + args.dataset_name + "_" + policy_exp_name
        sft_exp_name = 'sft_' + args.base_model_name + '_' + args.dataset_name
        policy_exp_names_full.append(policy_exp_name_full) 
        data_merger = DataMerge(policy_exp_name=policy_exp_name_full, sft_exp_name=sft_exp_name, merge_sft=True, eos_str=args.eos_str)
        data_merger.run()

    #Score
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_policy_exp, args, policy_exp_name) for policy_exp_name in policy_exp_names]
        
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 获取任务结果（如果有）

    #draw
    args.policy_exp_name = str(policy_exp_names_full)
    drawer = FigureDraw(args)
    drawer.draw_rh_bystep()

def gemma_ultra_lab2():
    args = setup_argparse()
    args.base_model_name = 'gemma2-2b'
    args.eos_str = '<eos>'
    args.dataset_name = 'ultrafb_bin'
    args.loss_name = 'ppo'
    args.lab_name = 'lab2'
    policy_exp_names = ['sigmoid','PARref1','PARref3', 'PARref5', 'PARref8', 'PAR']
    args.display_name = str(['Sigmoid','PARref1','PARref3', 'PARref5', 'PARref8','PARref10'])
    policy_exp_names_full = []
    #Merge
    for policy_exp_name in policy_exp_names:
        policy_exp_name_full = args.loss_name + '_' + args.base_model_name + "_" + args.dataset_name + "_" + policy_exp_name
        sft_exp_name = 'sft_' + args.base_model_name + '_' + args.dataset_name
        policy_exp_names_full.append(policy_exp_name_full) 
        data_merger = DataMerge(policy_exp_name=policy_exp_name_full, sft_exp_name=sft_exp_name, merge_sft=True, eos_str=args.eos_str)
        data_merger.run()

    #Score
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_policy_exp, args, policy_exp_name) for policy_exp_name in policy_exp_names]
        
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 获取任务结果（如果有）

    #draw
    args.policy_exp_name = str(policy_exp_names_full)
    drawer = FigureDraw(args)
    drawer.draw_rh_bystep() 

def gemma_ultra_lab34():
    args = setup_argparse()
    args.base_model_name = 'gemma2-2b'
    args.eos_str = '<eos>'
    args.dataset_name = 'ultrafb_bin'
    args.loss_name = 'ppo'
    args.lab_name = 'lab3-4'
    policy_exp_names = ['ceil5.0','ceil4.0','ceil3.0', 'kl0.01','kl0.02','kl0.05','kl0.1','WARMep2', 'minmaxep2','PARep2']
    args.display_name = str(['ceil5.0','ceil4.0','ceil3.0', 'kl0.01','kl0.02','kl0.05','kl0.1', 'WARM', 'Minmax','PAR'])
    policy_exp_names_full = []
    #Merge
    for policy_exp_name in policy_exp_names:
        policy_exp_name_full = args.loss_name + '_' + args.base_model_name + "_" + args.dataset_name + "_" + policy_exp_name
        sft_exp_name = 'sft_' + args.base_model_name + '_' + args.dataset_name
        policy_exp_names_full.append(policy_exp_name_full) 
        data_merger = DataMerge(policy_exp_name=policy_exp_name_full, sft_exp_name=sft_exp_name, merge_sft=True, eos_str=args.eos_str)
        data_merger.run()

    #Score
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_policy_exp, args, policy_exp_name) for policy_exp_name in policy_exp_names]
        
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 获取任务结果（如果有）

    #draw
    args.policy_exp_name = str(policy_exp_names_full)
    drawer = FigureDraw(args)
    drawer.draw_rh_bystep() 

def gemma_ultra_lab5():
    args = setup_argparse()
    args.base_model_name = 'gemma2-2b'
    args.eos_str = '<eos>'
    args.dataset_name = 'ultrafb_bin'
    args.loss_name = 'ppo'
    args.lab_name = 'lab5'
    policy_exp_names = ['relatanh', 'puretanh', 'relafittedpoly', 'purefittedpoly', 'PAR', 'sigmoid', 'relasigmoidk2', 'puresigmoidk2', 'relasigmoidk3', 'puresigmoidk3']
    args.display_name = str(['tanh(centered)', 'tanh(uncentered)', 'fittedpoly(centered)', 'fittedpoly(uncenetered)', 'sigmoid(centered)', 'sigmoid(uncentered)', 'sigmoidk2(centered)', 'sigmoidk2(uncentered)', 'sigmoidk3(centered)', 'sigmoidk3(uncentered)'])
    policy_exp_names_full = []
    #Merge
    for policy_exp_name in policy_exp_names:
        policy_exp_name_full = args.loss_name + '_' + args.base_model_name + "_" + args.dataset_name + "_" + policy_exp_name
        sft_exp_name = 'sft_' + args.base_model_name + '_' + args.dataset_name
        policy_exp_names_full.append(policy_exp_name_full) 
        data_merger = DataMerge(policy_exp_name=policy_exp_name_full, sft_exp_name=sft_exp_name, merge_sft=True, eos_str=args.eos_str)
        data_merger.run()

    #Score
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_policy_exp, args, policy_exp_name) for policy_exp_name in policy_exp_names]
        
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 获取任务结果（如果有）

    #draw
    args.policy_exp_name = str(policy_exp_names_full)
    drawer = FigureDraw(args)
    drawer.draw_rh_bystep() 

def gemma_hhrlhf_lab1():
    args = setup_argparse()
    args.base_model_name = 'gemma2-2b'
    args.eos_str = '<eos>'
    args.dataset_name = 'hh_rlhf'
    args.loss_name = 'ppo'
    args.lab_name = 'lab1'
    policy_exp_names = ['vanilla', 'WARM', 'ODIN', 'reg','meanstd', 'clip', 'minmax', 'lsc', 'PAR']
    args.display_name = str(['Vanilla', 'WARM', 'ODIN', 'Reg','Meanstd', 'Clip', 'Minmax', 'LSC', 'PAR'])
    policy_exp_names_full = []
    #Merge
    for policy_exp_name in policy_exp_names:
        policy_exp_name_full = args.loss_name + '_' + args.base_model_name + "_" + args.dataset_name + "_" + policy_exp_name
        sft_exp_name = 'sft_' + args.base_model_name + '_' + args.dataset_name
        policy_exp_names_full.append(policy_exp_name_full) 
        data_merger = DataMerge(policy_exp_name=policy_exp_name_full, sft_exp_name=sft_exp_name, merge_sft=True, eos_str=args.eos_str)
        data_merger.run()

    #Score
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_policy_exp, args, policy_exp_name) for policy_exp_name in policy_exp_names]
        
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 获取任务结果（如果有）

    #draw
    args.policy_exp_name = str(policy_exp_names_full)
    drawer = FigureDraw(args)
    drawer.draw_rh_bystep()

def llama_ultra_lab1():
    args = setup_argparse()
    args.base_model_name = 'llama3-8b'
    args.eos_str = '<|end_of_text|>'
    args.dataset_name = 'ultrafb_bin'
    args.loss_name = 'ppo'
    args.lab_name = 'lab1'
    policy_exp_names = ['vanilla', 'WARM', 'ODIN', 'reg','meanstd', 'clip', 'minmax', 'lsc', 'PAR']
    args.display_name = str(['Vanilla', 'WARM', 'ODIN', 'Reg','Meanstd', 'Clip', 'Minmax', 'LSC', 'PAR'])
    policy_exp_names_full = []

    #Merge
    for policy_exp_name in policy_exp_names:
        policy_exp_name_full = args.loss_name + '_' + args.base_model_name + "_" + args.dataset_name + "_" + policy_exp_name
        sft_exp_name = 'sft_' + args.base_model_name + '_' + args.dataset_name
        policy_exp_names_full.append(policy_exp_name_full) 
        data_merger = DataMerge(policy_exp_name=policy_exp_name_full, sft_exp_name=sft_exp_name, merge_sft=True, eos_str=args.eos_str)
        data_merger.run()

    #Score
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_policy_exp, args, policy_exp_name) for policy_exp_name in policy_exp_names]
        
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 获取任务结果（如果有）

    #draw
    args.policy_exp_name = str(policy_exp_names_full)
    drawer = FigureDraw(args)
    drawer.draw_rh_bystep()

def llama_hhrlhf_lab1():
    args = setup_argparse()
    args.base_model_name = 'llama3-8b'
    args.eos_str = '<|end_of_text|>'
    args.dataset_name = 'hh_rlhf'
    args.loss_name = 'ppo'
    args.lab_name = 'lab1'
    policy_exp_names = ['vanilla', 'WARM', 'ODIN', 'reg','meanstd', 'clip', 'minmax', 'lsc', 'PAR']
    args.display_name = str(['Vanilla', 'WARM', 'ODIN', 'Reg','Meanstd', 'Clip', 'Minmax', 'LSC', 'PAR'])
    policy_exp_names_full = []

    #Merge
    for policy_exp_name in policy_exp_names:
        policy_exp_name_full = args.loss_name + '_' + args.base_model_name + "_" + args.dataset_name + "_" + policy_exp_name
        sft_exp_name = 'sft_' + args.base_model_name + '_' + args.dataset_name
        policy_exp_names_full.append(policy_exp_name_full) 
        data_merger = DataMerge(policy_exp_name=policy_exp_name_full, sft_exp_name=sft_exp_name, merge_sft=True, eos_str=args.eos_str)
        data_merger.run()

    #Score
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_policy_exp, args, policy_exp_name) for policy_exp_name in policy_exp_names]
        
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 获取任务结果（如果有）

    #draw
    args.policy_exp_name = str(policy_exp_names_full)
    drawer = FigureDraw(args)
    drawer.draw_rh_bystep()

if __name__ == '__main__':
    gemma_ultra_lab1()
    # gemma_ultra_lab1_grpo()
    # gemma_ultra_lab2()
    # gemma_ultra_lab34()
    # gemma_ultra_lab5()
    # llama_ultra_lab1()
    # gemma_hhrlhf_lab1()
    # llama_hhrlhf_lab1()
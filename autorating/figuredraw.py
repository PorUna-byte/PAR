import json
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import numpy as np

# Interpolate missing values for proxy rewards and winrates
from scipy.interpolate import interp1d

class FigureDraw:
    """Draw figures for proxy/gold reward on different training steps
    """
    def __init__(self, config):
        self.base_model_name = config.base_model_name
        self.llm_model_name = config.llm_model_name
        self.dataset_name = config.dataset_name
        self.cache_dir = '/data/models'
        try:
            self.policy_exp_names = eval(config.policy_exp_name)
            self.policy_sample_dirs = [os.path.join(self.cache_dir, policy_exp_name, 'sample_on_test') for policy_exp_name in self.policy_exp_names]
        except Exception as e:
            pass

        self.step_to_policy_proxyreward = {}
        self.step_to_policy_winrate = {}
        self.kl_to_policy_proxyreward = {}
        self.kl_to_policy_winrate = {}

    def sigmoid(self, x):
        return 1/(1+math.exp(-x))
    
    def calculate_proxy_gold_score_bystep(self):
        """Calculate Winrate, policy-proxy_reward for each step"""
        for policy_sample_dir in self.policy_sample_dirs:
            self.step_to_policy_proxyreward[policy_sample_dir] = {}
            self.step_to_policy_winrate[policy_sample_dir] = {}

            for sub_dir in os.listdir(policy_sample_dir):
                data_path = os.path.join(policy_sample_dir, sub_dir, 'merged.json')
                data_dict = json.load(open(data_path, 'r'))
                if f'policy-{self.llm_model_name}_score' not in data_dict[0]:
                    break
                step = int(sub_dir[5:].strip())
                policy_proxy_reward = None
                if 'policy-proxy_reward' in data_dict[0]:
                    policy_proxy_reward = sum([item['policy-proxy_reward'] for item in data_dict])/len(data_dict)

                policy_win_counts = 0
                for item in data_dict:
                    if item[f'policy-{self.llm_model_name}_score']>item[f'sft-{self.llm_model_name}_score']:
                        policy_win_counts +=  1
                    elif item[f'policy-{self.llm_model_name}_score']==item[f'sft-{self.llm_model_name}_score']:
                        policy_win_counts += 0.5
                    
                policy_winrate = policy_win_counts/len(data_dict)
                self.step_to_policy_proxyreward[policy_sample_dir][step] = policy_proxy_reward
                self.step_to_policy_winrate[policy_sample_dir][step] = policy_winrate

    def calculate_proxy_gold_score_bykl(self):
        """Calculate Winrate, policy-proxy_reward for each step"""
        for policy_sample_dir in self.policy_sample_dirs:
            self.kl_to_policy_proxyreward[policy_sample_dir] = {}
            self.kl_to_policy_winrate[policy_sample_dir] = {}

            data_list = []
            for sub_dir in os.listdir(policy_sample_dir):
                data_path = os.path.join(policy_sample_dir, sub_dir, 'merged.json')
                data_dict = json.load(open(data_path, 'r'))
                if f'policy-{self.llm_model_name}_score' not in data_dict[0]:
                    break
                data_list.extend(data_dict)

            # Define the number of intervals
            num_intervals = 20  # Adjust as needed
            kl_min, kl_max = 0.0, 0.4
            interval_size = (kl_max - kl_min) / num_intervals

            # Automatically generate KL ranges
            kl_ranges = [(round(kl_min + i * interval_size, 4), round(kl_min + (i + 1) * interval_size, 4)) for i in range(num_intervals)]

            # Use a dictionary to store results for each range
            self.kl_to_policy_winrate[policy_sample_dir] = {start: [] for start, end in kl_ranges}
            self.kl_to_policy_proxyreward[policy_sample_dir] ={start: [] for start, end in kl_ranges}

            # Fast pinpointing using binary search-like mapping
            for item in data_list:
                kl_value = item.get("KL_distance", 0)
                index = math.floor((kl_value - kl_min) / interval_size)
                if 0 <= index < num_intervals:  # Ensure it stays within bounds
                    start, end = kl_ranges[index]
                    self.kl_to_policy_proxyreward[policy_sample_dir][start].append(item['policy-proxy_reward'])
                    win_count = 0
                    if item[f'policy-{self.llm_model_name}_score']>item[f'sft-{self.llm_model_name}_score']:
                        win_count =  1
                    elif item[f'policy-{self.llm_model_name}_score']==item[f'sft-{self.llm_model_name}_score']:
                        win_count += 0.5

                    self.kl_to_policy_winrate[policy_sample_dir][start].append(win_count)


    def plot_combined_bystep(self):

        pdf_filename =f"{self.base_model_name}_{self.dataset_name}_curve_bystep.pdf"
        # Create a figure and primary axis
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Secondary axis sharing the same x-axis
        ax2 = ax1.twinx()

        # Define color map for different loss functions
        color_map = plt.cm.get_cmap('tab10')
        loss_names = list(self.step_to_policy_winrate.keys())
        num_losses = len(loss_names)

        # Lists to store lines and labels for the combined legend
        lines = []
        labels = []

        for idx, loss_name in enumerate(loss_names):
            # Prepare data for proxy_reward
            step_proxyreward_dict = self.step_to_policy_proxyreward[loss_name]
            steps = sorted(step_proxyreward_dict.keys())
            proxy_rewards = [step_proxyreward_dict[step] for step in steps]

            # Prepare data for winrate
            step_winrate_dict = self.step_to_policy_winrate[loss_name]
            winrates = [step_winrate_dict[step] for step in steps]

            proxy_rewards_interp = interp1d(steps, proxy_rewards, kind='linear', fill_value="extrapolate")
            winrates_interp = interp1d(steps, winrates, kind='linear', fill_value="extrapolate")
            
            # Define a dense range of kls for smooth curves
            dense_steps = np.linspace(min(steps), max(steps), 40)

            # Interpolated values
            proxy_rewards_dense = proxy_rewards_interp(dense_steps)
            winrates_dense = winrates_interp(dense_steps)
        
            # Assign a unique color for each loss function
            color = color_map(idx / num_losses)

            # Plot proxy_reward on the primary axis
            if proxy_rewards[0]!=None:
                line1, = ax1.plot(dense_steps, proxy_rewards_dense, marker='o', linestyle='-', color=color, label=f'{loss_name} Proxy Reward')
                lines.append(line1)
                labels.append(f"{loss_name.split('/')[-2].split('_')[-1]} Proxy Reward")

            # Plot winrate on the secondary axis
            line2, = ax2.plot(dense_steps, winrates_dense, marker='x', linestyle='--', color=color, label=f'{loss_name} Winrate')
            lines.append(line2)
            labels.append(f"{loss_name.split('/')[-2].split('_')[-1]} Winrate")

        # Set axis labels and titles
        ax1.set_xlabel('Steps', fontsize=8)
        ax1.set_ylabel('Proxy Reward', fontsize=14, color='tab:blue')
        ax2.set_ylabel('Winrate', fontsize=14, color='tab:red')
        plt.title('Proxy Reward and Winrate vs. Steps for Each Loss Function', fontsize=16)

        # Set tick parameters
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Add grid to the primary axis
        ax1.grid(True)

        # Add combined legend
        ax1.legend(lines, labels, loc='lower right', fontsize=12)

        # Adjust layout
        plt.tight_layout()
        # Save the figure to PDF
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig)  # Save the figure to the PDF
            plt.close(fig)    # Close the figure

        print(f"Plot has been successfully saved to {pdf_filename}")
    
    def plot_combined_bykl(self):

        pdf_filename =f"{self.base_model_name}_{self.dataset_name}_curve_bykl.pdf"
        # Create a figure and primary axis
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Secondary axis sharing the same x-axis
        ax2 = ax1.twinx()

        # Define color map for different loss functions
        color_map = plt.cm.get_cmap('tab10')
        loss_names = list(self.kl_to_policy_winrate.keys())
        num_losses = len(loss_names)

        # Lists to store lines and labels for the combined legend
        lines = []
        labels = []

        for idx, loss_name in enumerate(loss_names):
            # Prepare data for proxy_reward
            kl_proxyreward_dict = self.kl_to_policy_proxyreward[loss_name]
            kl_winrate_dict = self.kl_to_policy_winrate[loss_name]
            kls = sorted(kl_proxyreward_dict.keys())
            valid_kls = []
            winrates = []
            proxy_rewards = []

            for kl in kls:
                if len(kl_proxyreward_dict[kl])>10:
                    valid_kls.append(kl)
                    proxy_rewards.append(sum(kl_proxyreward_dict[kl])/len(kl_proxyreward_dict[kl]))
                    winrates.append(sum(kl_winrate_dict[kl])/len(kl_winrate_dict[kl]))

            proxy_rewards_interp = interp1d(valid_kls, proxy_rewards, kind='linear', fill_value="extrapolate")
            winrates_interp = interp1d(valid_kls, winrates, kind='linear', fill_value="extrapolate")
            
            # Define a dense range of kls for smooth curves
            dense_kls = np.linspace(min(valid_kls), max(valid_kls), 40)

            # Interpolated values
            proxy_rewards_dense = proxy_rewards_interp(dense_kls)
            winrates_dense = winrates_interp(dense_kls)

            # Assign a unique color for each loss function
            color = color_map(idx / num_losses)
            
            # Plot proxy_reward on the primary axis
            if proxy_rewards[0]!=None:
                line1, = ax1.plot(dense_kls, proxy_rewards_dense, marker='o', linestyle='-', color=color, label=f'{loss_name} Proxy Reward')
                lines.append(line1)
                labels.append(f"{loss_name.split('/')[3]} Proxy Reward")

            # Plot winrate on the secondary axis
            line2, = ax2.plot(dense_kls, winrates_dense, marker='x', linestyle='--', color=color, label=f'{loss_name} Winrate')
            lines.append(line2)
            labels.append(f"{loss_name.split('/')[3]} Winrate")

        # Set axis labels and titles
        ax1.set_xlabel('KLs', fontsize=8)
        ax1.set_ylabel('Proxy Reward', fontsize=14, color='tab:blue')
        ax2.set_ylabel('Winrate', fontsize=14, color='tab:red')
        plt.title('Proxy Reward and Winrate vs. Steps for Each Loss Function', fontsize=16)

        # Set tick parameters
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Add grid to the primary axis
        ax1.grid(True)

        # Add combined legend
        ax1.legend(lines, labels, loc='lower right', fontsize=12)

        # Adjust layout
        plt.tight_layout()
        # Save the figure to PDF
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig)  # Save the figure to the PDF
            plt.close(fig)    # Close the figure

        print(f"Plot has been successfully saved to {pdf_filename}")

    def draw_gold_reward_acc(self):
        """Calculate the accuracy for gold_reward on test set, The accuracy is defined as the fraction of chosen-gold_reward is higher than rejected-gold_reward"""
        score_list = json.load(open(f'/data/data/{self.dataset_name}/{self.llm_model_name}_score_ontest.json', 'r'))
        acc = 0
        for item in score_list:
            if item[f'chosen-{self.llm_model_name}_score']>item[f'rejected-{self.llm_model_name}_score']:
                acc += 1
            elif item[f'chosen-{self.llm_model_name}_score']==item[f'rejected-{self.llm_model_name}_score']:
                acc += 0.5
        
        acc /= len(score_list)
        with open(f'{self.llm_model_name}_testacc.txt', 'a') as f:
            f.write(f'The accuracy of gold reward given by {self.llm_model_name} on the test set of {self.dataset_name} is {acc}\n')

    def draw_rh_bystep(self):
        self.calculate_proxy_gold_score_bystep()
        self.plot_combined_bystep()

    def draw_rh_bykl(self):
        self.calculate_proxy_gold_score_bykl()
        self.plot_combined_bykl()
import json
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import numpy as np
import json

# Interpolate missing values for proxy rewards and winrates
from scipy.interpolate import interp1d
from utils.utils import sigmoid

class FigureDraw:
    """Draw figures for proxy/gold reward on different training steps
    """
    def __init__(self, config):
        self.base_model_name = config.base_model_name
        self.llm_model_name = config.llm_model_name
        self.dataset_name = config.dataset_name
        self.lab_name = config.lab_name
        if self.dataset_name == 'ultrafb_bin':
            self.test_set = json.load(open(f"/data/data/{self.dataset_name}/{self.base_model_name}_test_prefs_plus.json"))
        elif self.dataset_name == 'hh_rlhf':
            self.test_set = json.load(open(f"/data/data/hh-rlhf-helpful/{self.base_model_name}_test_prefs_plus.json"))

        self.prompt2refreward = {item['prompt']:item['sample_rewards'] for item in self.test_set}

        self.cache_dir = '/data/models'
        self.dir2display = {}
        try:
            self.policy_exp_names = eval(config.policy_exp_name)
            self.display_name = eval(config.display_name)
            self.policy_sample_dirs = [os.path.join(self.cache_dir, policy_exp_name, 'sample_on_test') for policy_exp_name in self.policy_exp_names]
            for idx, policy_sample_dir in enumerate(self.policy_sample_dirs):
                self.dir2display[policy_sample_dir] = self.display_name[idx]
        except Exception as e:
            pass

        self.step_to_policy_proxyreward = {}
        self.step_to_policy_winrate = {}
        self.step_to_preference_score = {}

        self.kl_to_policy_proxyreward = {}
        self.kl_to_policy_winrate = {}

    def sigmoid(self, x):
        return 1/(1+math.exp(-x))
    
    def calculate_proxy_gold_score_bystep(self):
        """Calculate Winrate, policy-proxy_reward for each step"""
        for policy_sample_dir in self.policy_sample_dirs:

            self.step_to_policy_proxyreward[policy_sample_dir] = {}
            self.step_to_policy_winrate[policy_sample_dir] = {}
            self.step_to_preference_score[policy_sample_dir] = {}

            for sub_dir in os.listdir(policy_sample_dir):
                data_path = os.path.join(policy_sample_dir, sub_dir, 'merged.json')
                data_dict = json.load(open(data_path, 'r'))
                if f'policy-{self.llm_model_name}_score' not in data_dict[0]:
                    break
                step = int(sub_dir[5:].strip())

                policy_proxy_reward = sum([item['policy-proxy_reward'] for item in data_dict])/len(data_dict)

                policy_preference_score = 0
                miss = 0
                for item in data_dict:
                    clean_prompt = item['prompt'].replace("<|user|>", "").replace("<|assistant|>","")
                    if clean_prompt not in self.prompt2refreward:
                        miss+=1
                        continue
                    refrewards = self.prompt2refreward[clean_prompt]
                    item_preference_score = sum([sigmoid(item['policy-proxy_reward']-refreward) for refreward in refrewards])/len(refrewards)
                    policy_preference_score += item_preference_score
                policy_preference_score /= (len(data_dict)-miss)
                print(f'miss:{miss}\n')

                policy_win_counts = 0
                for item in data_dict:
                    if item[f'policy-{self.llm_model_name}_score']>item[f'sft-{self.llm_model_name}_score']:
                        policy_win_counts +=  1
                    elif item[f'policy-{self.llm_model_name}_score']==item[f'sft-{self.llm_model_name}_score']:
                        policy_win_counts += 0.5
                policy_winrate = policy_win_counts/len(data_dict)

                self.step_to_policy_proxyreward[policy_sample_dir][step] = policy_proxy_reward
                self.step_to_policy_winrate[policy_sample_dir][step] = policy_winrate
                self.step_to_preference_score[policy_sample_dir][step] = policy_preference_score

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


    def plot_proxy_gold_score_bystep(self):
        pdf_filename = f"{self.base_model_name}_{self.dataset_name}_curve_bystep_{self.lab_name}.pdf"
        
        # Create a figure and primary axis
        fig, ax1 = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#f7f7f7')  # Set background color for the figure
        ax1.set_facecolor('#ffffff')  # Set background color for the plot area

        # Secondary axis sharing the same x-axis
        ax2 = ax1.twinx()

        # Define color map for different loss functions
        color_map = plt.cm.get_cmap('tab20')  # Use a more visually appealing colormap
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

            # Interpolate data for smooth curves
            proxy_rewards_interp = interp1d(steps, proxy_rewards, kind='linear', fill_value="extrapolate")
            winrates_interp = interp1d(steps, winrates, kind='linear', fill_value="extrapolate")
            
            # Define a dense range of steps for smooth curves
            dense_steps = np.linspace(min(steps), max(steps), 10)

            # Interpolated values
            proxy_rewards_dense = proxy_rewards_interp(dense_steps)
            winrates_dense = winrates_interp(dense_steps)
        
            # Assign a unique color for each loss function
            color = color_map(idx % num_losses)  # Use modulo to avoid color repetition

            # Determine linewidth based on loss_name
            linewidth = 4 if 'PAR' in loss_name else 2  # Set thicker line for PAR

            # Plot proxy_reward on the primary axis
            if proxy_rewards[0] is not None:
                line1, = ax1.plot(
                    dense_steps, proxy_rewards_dense, 
                    marker='o', markersize=6, linestyle='-', linewidth=linewidth,  # Use linewidth here
                    color=color, alpha=0.8, label=f'{loss_name} Proxy Reward'
                )
                lines.append(line1)
                labels.append(self.dir2display[loss_name])  # Simplified label

            # Plot winrate on the secondary axis
            line2, = ax2.plot(
                dense_steps, winrates_dense, 
                marker='x', markersize=6, linestyle='--', linewidth=linewidth,  # Use linewidth here
                color=color, alpha=0.8, label=f'{loss_name} Winrate'
            )
            lines.append(line2)

        # Set axis labels and titles
        ax1.set_xlabel('Steps', fontsize=18, labelpad=10)
        ax1.set_ylabel('Proxy Reward', fontsize=18, color='tab:blue', labelpad=10)
        ax2.set_ylabel('Winrate', fontsize=18, color='tab:red', labelpad=10)
        plt.title('Proxy Reward and Winrate vs. Steps', fontsize=18, pad=20)

        # Set tick parameters
        ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=18)
        ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=18)
        ax1.tick_params(axis='x', labelsize=18)

        # Add grid to the primary axis
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Add combined legend (one entry per method)
        ax1.legend(
            lines[::2], labels,  # Use every other line to avoid duplicate entries
            loc='upper left',  # Move legend to upper left
            bbox_to_anchor=(1.05, 0.4),  # Place legend outside the plot
            fontsize=12, 
            frameon=True, framealpha=0.9, facecolor='white',
            ncol=1  # Use one columns for the legend
        )

        # Adjust layout to make space for the legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)  # Adjust right margin to make space for the legend

        # Save the figure to PDF
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig, bbox_inches='tight')  # Save the figure to the PDF
            plt.close(fig)  # Close the figure

        print(f"Plot has been successfully saved to {pdf_filename}")
    
    def plot_preference_score_vs_winrate(self):
        pdf_filename = f"{self.base_model_name}_{self.dataset_name}_preference_vs_winrate_{self.lab_name}.pdf"
        
        # Create a figure and primary axis
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#f7f7f7')  # Set background color for the figure
        ax.set_facecolor('#ffffff')  # Set background color for the plot area

        # Define color map for different loss functions
        color_map = plt.cm.get_cmap('tab10')  # Use a visually appealing colormap
        loss_names = list(self.step_to_policy_winrate.keys())
        num_losses = len(loss_names)

        # Lists to store lines and labels for the legend
        lines = []
        labels = []

        # Collect all preference scores to determine the range for the perfect calibration line
        all_preference_scores = []
        for loss_name in loss_names:
            step_preference_dict = self.step_to_preference_score[loss_name]
            all_preference_scores.extend(step_preference_dict.values())

        # Plot the perfect calibration line (slope=1)
        perfect_line, = ax.plot(
            [0, 1], [0, 1], 
            linestyle='--', linewidth=2, color='gray', alpha=0.8, 
            label='Perfect Calibration'
        )
        lines.append(perfect_line)
        labels.append('Perfect Calibration')

        for idx, loss_name in enumerate(loss_names):
            # Prepare data for winrate
            step_winrate_dict = self.step_to_policy_winrate[loss_name]
            steps = sorted(step_winrate_dict.keys())
            winrates = [step_winrate_dict[step] for step in steps]

            # Prepare data for preference_score
            step_preference_dict = self.step_to_preference_score[loss_name]
            preference_scores = [step_preference_dict[step] for step in steps]

            # Assign a unique color for each loss function
            color = color_map(idx % num_losses)  # Use modulo to avoid color repetition

            # Plot preference_score vs. winrate
            line, = ax.plot(
                preference_scores, winrates, 
                marker='o', markersize=6, linestyle='-', linewidth=2, 
                color=color, alpha=0.8, label=f'{loss_name}'
            )
            lines.append(line)
            labels.append(self.dir2display[loss_name])  # Simplified label

        # Set axis labels and title
        ax.set_xlabel('Preference Score', fontsize=18, labelpad=10)
        ax.set_ylabel('Winrate', fontsize=18, labelpad=10)
        plt.title('Winrate vs. Preference Score', fontsize=18, pad=20)

        # Set tick parameters
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        # Add grid to the plot
        ax.grid(True, linestyle='--', alpha=0.6)

        # Add legend
        ax.legend(
            lines, labels,  # Use lines and labels for the legend
            loc='lower left',  # Move legend to upper left
            bbox_to_anchor=(1.02, 0),  # Place legend outside the plot
            fontsize=12, 
            frameon=True, framealpha=0.9, facecolor='white',
            ncol=1  # Use one column for the legend
        )

        # Adjust layout to make space for the legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.7)  # Adjust right margin to make space for the legend

        # Save the figure to PDF
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig, bbox_inches='tight')  # Save the figure to the PDF
            plt.close(fig)  # Close the figure

        print(f"Plot has been successfully saved to {pdf_filename}")


    def plot_winrate_histogram(self):
        pdf_filename = f"{self.base_model_name}_{self.dataset_name}_winrate_histogram_{self.lab_name}.pdf"
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#f7f7f7')  # Set background color for the figure
        ax.set_facecolor('#ffffff')  # Set background color for the plot area

        # Define color map for different loss functions
        color_map = plt.cm.get_cmap('tab10')  # Use a visually appealing colormap
        loss_names = list(self.step_to_policy_winrate.keys())
        
        num_losses = len(loss_names)

        # Calculate the average winrate for each method
        avg_winrates = []
        for loss_name in loss_names:
            step_winrate_dict = self.step_to_policy_winrate[loss_name]
            winrates = [step_winrate_dict[step] for step in step_winrate_dict.keys()]
            avg_winrate = np.mean(winrates)
            avg_winrates.append(avg_winrate)

        # Assign a unique color for each method
        colors = [color_map(i % num_losses) for i in range(num_losses)]

        # Plot the histogram
        label_names = [self.dir2display[loss_name] for loss_name in loss_names]
        bars = ax.bar(label_names, avg_winrates, color=colors, alpha=0.8)

        # Set axis labels and title
        ax.set_xlabel('Methods', fontsize=18, labelpad=10)
        ax.set_ylabel('Average Winrate', fontsize=18, labelpad=10)
        plt.title('Average Winrate', fontsize=18, pad=20)

        # Set tick parameters
        ax.tick_params(axis='x', labelsize=18, rotation=45)  # Rotate x-axis labels for better readability
        ax.tick_params(axis='y', labelsize=18)

        # Add grid to the axis
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=18)

        # Adjust layout to make space for the labels
        plt.tight_layout()

        # Save the figure to PDF
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig, bbox_inches='tight')  # Save the figure to the PDF
            plt.close(fig)  # Close the figure

        print(f"Histogram has been successfully saved to {pdf_filename}")

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
        self.plot_proxy_gold_score_bystep()
        self.plot_preference_score_vs_winrate()
        self.plot_winrate_histogram()


    def draw_rh_bykl(self):
        self.calculate_proxy_gold_score_bykl()
        self.plot_combined_bykl()
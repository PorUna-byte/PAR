import torch
from utils.utils import sigmoid, mean, std
from scipy.stats import norm

class RunningStats:
    def __init__(self):
        self.n = 0  # Total number of rewards observed
        self.mean = 0.0  # Running mean
        self.M2 = 0.0  # Sum of squared differences from the mean
        self.max = -10000
        self.min = 10000

    def update(self, rewards):
        """
        Update running statistics with a tensor of rewards.
        Args:
            rewards (torch.Tensor): A tensor of shape (batch_size,).
        """
        batch_size = rewards.shape[0]
        batch_mean = rewards.mean().item()  # Mean of the current batch
        batch_var = rewards.var(unbiased=False).item()  # Variance of the current batch
        batch_max = rewards.max().item()
        batch_min = rewards.min().item()

        self.max = max(self.max, batch_max)
        self.min = min(self.min, batch_min)

        # Update the total count
        new_n = self.n + batch_size

        # Compute the new mean
        delta = batch_mean - self.mean
        self.mean += delta * (batch_size / new_n)

        # Update M2 (sum of squared differences from the mean)
        self.M2 += batch_var * batch_size + delta**2 * (self.n * batch_size / new_n)

        # Update the count
        self.n = new_n

    def normalize(self, reward):
        mean = self.get_mean()
        std = self.get_std()
        return (reward - mean) / (std if std > 0 else 1.0)
    
    def clip(self, reward):
        mean = self.get_mean()
        std = self.get_std()
        return torch.clip(reward, mean-std, mean+std)
    
    def minmax(self, reward):
        return (reward-self.min)/(self.max-self.min)
    
    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.M2 / self.n if self.n > 1 else 0.0

    def get_std(self):
        return (self.get_variance())**0.5

class RewardShaper:
    """A basic class to shape reward for a raw reward 'reward'"""
    def __init__(self, config):
        self.config = config
        self.runningstats = RunningStats()
    
    def calculate_relative_prefs(self, reward, sftref_rewards):
        prefs = 0
        for rr in sftref_rewards:
            prefs += sigmoid(reward-rr)
        
        return prefs/len(sftref_rewards)

    def calculate_relative_wins(self, reward, sftref_rewards, max_num=5):
        win_count = 0
        for rr in sftref_rewards[:max_num]:
            win_count += 1 if reward>rr else 0
    
        return win_count/len(sftref_rewards[:max_num])
    

    def calculate_percentile(self, mean, std_dev, percentile):
        """
        Calculate the specified percentile of a given normal distribution.
        
        Parameters:
        mean (float): The mean of the normal distribution
        std_dev (float): The standard deviation of the normal distribution (square root of variance)
        percentile (float): The desired percentile, ranging between 0 and 1
        
        Returns:
        float: The value corresponding to the specified percentile
        """
        return norm.ppf(percentile, loc=mean, scale=std_dev+1e-5)
    
    def shaped_reward(self, reward, masks, sftref_rewards):
        batch_size = masks.shape[0]
        #Penalize longer response to prevent the model from hacking the length of response, default to be true
        if self.config.reward_penalize_length and not self.config.reward_odin:
            for row in range(batch_size):
                if masks[row].sum() > self.config.gen_valid_len:
                    reward[row] -= self.config.len_penalty*(masks[row].sum()-self.config.gen_valid_len)


        #Normalize the reward via the running mean and std
        if self.config.reward_meanstd:
            self.runningstats.update(reward)
            reward = self.runningstats.normalize(reward)  
        elif self.config.reward_clipping:
            self.runningstats.update(reward)
            reward = self.runningstats.clip(reward)
        #Normalize the reward via the running max and min
        elif self.config.reward_minmax:
            self.runningstats.update(reward)
            reward = self.runningstats.minmax(reward)

        #Normalize the reward via sigmoid
        elif self.config.reward_sigmoid:
            reward = torch.sigmoid(reward)
        elif self.config.reward_par:
            for row in range(batch_size):
                reward[row] = self.calculate_relative_prefs(reward[row], sftref_rewards[row])

        #Normalize the reward via log-sigmoid centered transformation
        elif self.config.reward_lsc:
            for row in range(batch_size):
                mean_val, std_val = mean(sftref_rewards[row]), std(sftref_rewards[row])
                ref_reward = self.calculate_percentile(mean_val, std_val, 0.85)
                reward[row] = torch.log(torch.sigmoid(reward[row]-ref_reward))

        #Normalize the reward via relative winrate   
        elif self.config.reward_discrete:
            for row in range(batch_size):
                reward[row] = self.calculate_relative_wins(reward[row], sftref_rewards[row], self.config.reward_discrete_maxref)
        
        return reward
    


import torch
from utils.utils import sigmoid, mean, std
from scipy.stats import norm
import numpy as np
from scipy.optimize import curve_fit

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
    
# 定义高阶多项式函数
def high_order_poly(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

class RewardShaper:
    """A basic class to shape reward for a raw reward 'reward'"""
    def __init__(self, config):
        self.config = config
        self.runningstats = RunningStats()
        # 生成 Sigmoid 数据用于拟合
        x_fit = np.linspace(-5, 5, 100)
        y_fit =  1 / (1 + np.exp(-x_fit))
        # 使用 curve_fit 进行多项式拟合
        params, _ = curve_fit(high_order_poly, x_fit, y_fit)
        self.poly_params = params

    def calculate_relative_sigmoid(self, reward, sftref_rewards):
        scores = 0
        for rr in sftref_rewards:
            scores += 1/(1+torch.exp(-self.config.sigmoid_k*(reward-rr)))
        
        return scores/len(sftref_rewards)
    
    def calculate_centered(self, reward, sftref_rewards):
        scores = 0
        for rr in sftref_rewards:
            scores += reward - rr
        
        return scores/len(sftref_rewards)

    def calculate_relative_tanh(self, reward, sftref_rewards):
        scores = 0
        for rr in sftref_rewards:
            scores += torch.tanh(reward-rr)
        
        return scores/len(sftref_rewards)

    def calculate_relative_arctan(self, reward, sftref_rewards):
        scores = 0
        for rr in sftref_rewards:
            scores += torch.atan(reward-rr)
        
        return scores/len(sftref_rewards)
    
    def calculate_relative_fitted_poly(self, reward, sftref_rewards):
        scores = 0
        for rr in sftref_rewards:
            scores += self.calculate_fitted_poly(reward-rr)
        return scores/len(sftref_rewards)
    
    # 定义新分段多项式函数
    def calculate_fitted_poly(self, x):
        return torch.where(
            x <= -5,
            0,
            torch.where(
                x >= 5,
                1,
                high_order_poly(x, *self.poly_params)
            )
        )
    
    def calculate_tanh(self, reward):
        return torch.tanh(reward)
        
    def calculate_arctan(self, reward):
        return torch.atan(reward)
    
    def calculate_sigmoid(self, reward):
        return 1/(1+torch.exp(-self.config.sigmoid_k*reward))
    
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
        if self.config.reward_ceil!=None:
            reward = torch.clamp(reward, max=self.config.reward_ceil)

        #Penalize longer response to prevent the model from hacking the length of response, default to be true
        if self.config.reward_penalize_length and not self.config.reward_odin:
            for row in range(batch_size):
                if masks[row].sum() > self.config.gen_valid_len:
                    reward[row] -= self.config.len_penalty*(masks[row].sum()-self.config.gen_valid_len)

        if self.config.reward_centered:
            for row in range(batch_size):
                reward[row] = self.calculate_centered(reward[row], sftref_rewards[row][:self.config.reward_maxref])

        if self.config.reward_sgfc:
            for row in range(batch_size):
                reward[row] = self.calculate_relative_sigmoid(reward[row]-3, sftref_rewards[row][:self.config.reward_maxref])

        #Normalize the reward via the running mean and std
        if self.config.reward_meanstd:
            self.runningstats.update(reward)
            reward = self.runningstats.normalize(reward)  
        if self.config.reward_clipping:
            self.runningstats.update(reward)
            reward = self.runningstats.clip(reward)
        if self.config.reward_ceil!=None:
            reward = torch.clamp(reward, max=self.config.reward_ceil)
        #Normalize the reward via the running max and min
        if self.config.reward_minmax:
            self.runningstats.update(reward)
            reward = self.runningstats.minmax(reward)

        #Normalize the reward via sigmoid
        if self.config.reward_sigmoid:
            if self.config.reward_relative:
                for row in range(batch_size):
                    reward[row] = self.calculate_relative_sigmoid(reward[row], sftref_rewards[row][:self.config.reward_maxref])
            else:
                reward = self.calculate_sigmoid(reward)

        if self.config.reward_tanh:
            if self.config.reward_relative:
                for row in range(batch_size):
                    reward[row] = self.calculate_relative_tanh(reward[row], sftref_rewards[row][:self.config.reward_maxref])
            else:
                reward = self.calculate_tanh(reward)

        if self.config.reward_fittedpoly:
            if self.config.reward_relative:
                for row in range(batch_size):
                    reward[row] = self.calculate_relative_fitted_poly(reward[row], sftref_rewards[row][:self.config.reward_maxref])
            else:
                reward = self.calculate_fitted_poly(reward)

        #Normalize the reward via log-sigmoid centered transformation
        if self.config.reward_lsc:
            for row in range(batch_size):
                mean_val, std_val = mean(sftref_rewards[row]), std(sftref_rewards[row])
                ref_reward = self.calculate_percentile(mean_val, std_val, 0.85)
                reward[row] = torch.log(torch.sigmoid(reward[row]-ref_reward))


        return reward
    


import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
policy_loss_df = pd.read_csv('ppo_gemma2-2b_ultrafb_bin_policy_loss.csv')
critic_loss_df = pd.read_csv('ppo_gemma2-2b_ultrafb_bin_critic_loss.csv')

# 提取数据
steps = policy_loss_df['Step']
vanilla_policy_loss = policy_loss_df['ppo_gemma2-2b_ultrafb_bin_vanilla - loss/policy']
par_policy_loss = policy_loss_df['ppo_gemma2-2b_ultrafb_bin_par - loss/policy']

vanilla_critic_loss = critic_loss_df['ppo_gemma2-2b_ultrafb_bin_vanilla - loss/critic']
par_critic_loss = critic_loss_df['ppo_gemma2-2b_ultrafb_bin_par - loss/critic']

# 设置全局字体大小
plt.rcParams.update({'font.size': 20})

# 绘制 policy loss 曲线
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(steps, vanilla_policy_loss, label='Vanilla Policy Loss', color='blue')
plt.plot(steps, par_policy_loss, label='PAR Policy Loss', color='red')
plt.xlabel('Steps', fontsize=25)
plt.ylabel('Policy Loss', fontsize=25)
plt.title('Policy Loss Comparison', fontsize=25)
plt.ylim(0, 0.1)  # 设置 y 轴范围为 0 到 2
plt.legend(fontsize=25)
plt.grid(True)

# 绘制 critic loss 曲线
plt.subplot(1, 2, 2)
plt.plot(steps, vanilla_critic_loss, label='Vanilla Critic Loss', color='blue')
plt.plot(steps, par_critic_loss, label='PAR Critic Loss', color='red')
plt.xlabel('Steps', fontsize=25)
plt.ylabel('Critic Loss', fontsize=25)
plt.title('Critic Loss Comparison', fontsize=25)
plt.legend(fontsize=25)
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()
plt.savefig("loss_figure.pdf")
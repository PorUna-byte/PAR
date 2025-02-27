# PAR
Welcome to the PAR open-source project! This project aims to provide a collection of advanced reinforcement learning algorithm implementations, including SFT, Reward Modeling, PPO, DPO, IPO, KTO, SLIC, ReMax, and GRPO. Our goal is to help researchers and developers easily apply and extend these algorithms.

We also explore some methods to mitigate reward hacking problem, such as WARM, ODIN, Reg, Meanstd, Clip, Minmax, LSC, PAR.
For more details, see this paper: <https://arxiv.org/abs/2502.18770>


## Directory Structure
```
RH_RESEARCH
├── auto_sh
├── autorating
├── benchmark
├── configs
├── dataloaders
├── figures
├── models
├── trainers
├── utils
├── .gitignore
├── LICENSE
├── README.md
└── train.py
```

## Project Overview

This project implements the following algorithms:

- **SFT (Supervised Fine-Tuning)**
- **Reward Modeling**
- **PPO (Proximal Policy Optimization)**
- **DPO (Direct Policy Optimization)**
- **IPO (A General Theoretical Paradigm to Understand Learning from Human Preferences)**
- **KTO (Kahneman-Tversky Optimization)**
- **SLiC-HF (Sequence Likelihood Calibration with Human Feedback)**
- **ReMax**
- **GRPO (Group Relative Policy Optimization)**

## Usage

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/PorUna-byte/PAR.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add secret.py into the utils directory:
    ```bash
    wanbkey = 'xxx'
    huggingface_token = 'xxx'
    deepseek_key = 'xxx'
    deepseek_base = 'https://api.deepseek.com'
    deepseek_model = 'deepseek-chat'
    ```
4. Run utils/download.py to download model and datasets
    ```bash
    cd utils
    python download.py
    ```
5. Run `train.py` to start training the model:
   ```bash
   #SFT Training 
   torchrun train.py --loss_name sft --model_name gemma2-2b --dataset ultrafb_bin --wandb_enabled --wandb_project sft --global_batch_size 64 --learning_rate 5e-6 --max_grad_norm 10.0 --sample_ontest --n_epoch 2 
   #Reward Training
   torchrun train.py --loss_name reward --model_name gemma2-2b  --dataset ultrafb_bin --wandb_enabled --wandb_project reward --global_batch_size 32 --learning_rate 5e-6 --max_grad_norm 5.0 --exp_name reward_gemma2-2b_ultrafb_bin --n_epoch 1 
   #PPO Training
   torchrun train.py --loss_name ppo --model_name gemma2-2b --dataset ultrafb_bin --wandb_enabled --wandb_project ppo --global_batch_size 40 --learning_rate 3e-7  --critic_lr 5e-6 --max_grad_norm 5.0  --policy_path /data/models/sft_gemma2-2b_ultrafb_bin --policy_tag latest_hf --reference_path /data/models/sft_gemma2-2b_ultrafb_bin --reference_tag latest_hf --reward_path /data/models/reward_gemma2-2b_ultrafb_bin     --reward_tag latest_hf  --critic_path /data/models/reward_gemma2-2b_ultrafb_bin         --critic_tag latest_hf  --exp_name ppo_gemma2-2b_ultrafb_bin_vanilla  --save_ckps all --n_epochs 1 
   ```
    #The usage of ReMax, GRPO, DPO, etc are similiar to PPO training.

6. Run 'autorating/rh_analyze.py' to leverage a LLM score the winrate
    ```bash
    cd autorating
    python rh_analyze.py
    ```

## Contribution Guidelines

We welcome contributions of any kind! If you have any suggestions or find any issues, please submit an issue or pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact Us

If you have any questions, please contact us at [fujy22@m.fudan.edu.cn](mailto:fujy22@m.fudan.edu.cn).

---

Thank you for your interest and support in the project! We look forward to your contributions and feedback.




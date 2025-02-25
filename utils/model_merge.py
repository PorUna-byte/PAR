import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.models import AutoModelForCausalLMWithScalarHead
import argparse

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_name', type=str, default='llama3-8b', help='The base model for Alignment')
    parser.add_argument('--dataset_name', type=str, default='ultrafb_bin', help='The dataset we are using')
    args = parser.parse_args()
    return args

class ModelMerger:
    def __init__(self, models):
        """
        Initialize the ModelMerger with a list of models.
        
        :param models: List of models (nn.Module instances) to be merged.
        """
        self.models = models
        self.merged_model = None

    def merge_models(self):
        """
        Merge the models by calculating the mean of their parameters.
        """
        # Ensure there are models to merge
        if not self.models:
            raise ValueError("No models provided for merging.")

        # Initialize the merged model with the same architecture as the first model
        self.merged_model = self.models[0]

        # Iterate over all parameters in the models
        for param_name, param in self.models[0].named_parameters():
            # Stack all parameters from the models
            stacked_params = torch.stack([model.state_dict()[param_name] for model in self.models])
            # Calculate the mean of the stacked parameters
            mean_param = torch.mean(stacked_params, dim=0)
            # Update the parameter in the merged model
            param.data.copy_(mean_param)

    def save_merged_model(self, path):
        """
        Save the merged model to a file.
        
        :param path: Path to save the merged model.
        """
        if self.merged_model is None:
            raise ValueError("No merged model available. Call merge_models() first.")
        torch.save(self.merged_model.state_dict(), path)




dtype = torch.bfloat16
args = setup_argparse()
if args.base_model_name=='gemma2-2b':
    base_model_path = "/data/models/Gemma-2-2B"
    flash_attention = "eager"
elif args.base_model_name=='llama3-8b':
    base_model_path = "/data/models/Llama-3-8B"
    flash_attention = None

# Example usage:
# Assuming model1, model2, and model3 are instances of the same model architecture
models = []
for i in range(1,6):
    model = AutoModelForCausalLMWithScalarHead.from_pretrained(base_model_path, low_cpu_mem_usage=True, attn_implementation=flash_attention, torch_dtype=dtype, trust_remote_code=True)
    reward_path = f"/data/models/reward_{args.base_model_name}_{args.dataset_name}_M{i}"
    state_dict = torch.load(os.path.join(reward_path, "latest_hf.pt"), weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(dtype)
    models.append(model)


# Create a ModelMerger instance with the models to merge
merger = ModelMerger(models)

# Merge the models
merger.merge_models()

# Save the merged model
os.makedirs(f"/data/models/reward_{args.base_model_name}_{args.dataset_name}_WARM/", exist_ok=True)
merger.save_merged_model(f"/data/models/reward_{args.base_model_name}_{args.dataset_name}_WARM/latest_hf.pt")
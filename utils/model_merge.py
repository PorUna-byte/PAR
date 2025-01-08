import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.models import AutoModelForCausalLMWithScalarHead

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



base_model_path = "/data/models/Gemma-2-2B"
flash_attention = "eager"
dtype = torch.bfloat16

# Example usage:
# Assuming model1, model2, and model3 are instances of the same model architecture
models = []
for i in range(1,6):
    model = AutoModelForCausalLMWithScalarHead.from_pretrained(base_model_path, low_cpu_mem_usage=True, attn_implementation=flash_attention, torch_dtype=dtype, trust_remote_code=True)
    reward_path = f"/data/models/reward_gemma2-2b_ultrafb_bin_M{i}"
    state_dict = torch.load(os.path.join(reward_path, "latest_hf.pt"), weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(dtype).to(f'cuda:0')
    models.append(model)


# Create a ModelMerger instance with the models to merge
merger = ModelMerger(models)

# Merge the models
merger.merge_models()

# Save the merged model
os.makedirs("/data/models/reward_gemma2-2b_ultrafb_bin_WARM/", exist_ok=True)
merger.save_merged_model("/data/models/reward_gemma2-2b_ultrafb_bin_WARM/latest_hf.pt")
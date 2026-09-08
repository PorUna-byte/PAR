# Copyright (c) 2024 Stepfun AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains the classes necessary for doing PPO (offline, one-step) with language model.
This code is largely from the TRL library, with some modifications to ensure stability.
"""

import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM
import os
import torch

class PreTrainedModelWrapper(nn.Module):
    r"""
    A wrapper class around a (`transformers.PreTrainedModel`) to be compatible with the
    (`~transformers.PreTrained`) class in order to keep some attributes and methods of the
    (`~transformers.PreTrainedModel`) class.

    Attributes:
        pretrained_model: (`transformers.PreTrainedModel`)
            The model to be wrapped.
        transformers_parent_class: (`transformers.PreTrainedModel`)
            The parent class of the model to be wrapped.
    """
    transformers_parent_class = None
    def __init__(self, pretrained_model=None):
        super().__init__()
        self.pretrained_model = pretrained_model

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *args, **kwargs):
        r"""
        Instantiates a new model from a pretrained model from `transformers`. The
        pretrained model is loaded using the `from_pretrained` method of the
        `transformers.PreTrainedModel` class. The arguments that are specific to the
        `transformers.PreTrainedModel` class are passed along this method and filtered
        out from the `kwargs` argument.

        Args:
            pretrained_model_path (`str`):
                The path to the pretrained model.
            **args (`tuple`, *optional*):
                Additional arguments passed along to the underlying model's
                `from_pretrained` method. 
            **kwargs (`dict`, *optional*):
                Additional keyword arguments passed along to the underlying model's
                `from_pretrained` method. 
        """

        # First, load the pre-trained model using the parent-class
        # either `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`
        if isinstance(pretrained_model_path, str):
            pretrained_model = cls.transformers_parent_class.from_pretrained(
                pretrained_model_path, *args, **kwargs
            )
        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string, "
                f"but is {type(pretrained_model_path)}"
            )
        
        # call the __init__ method to create an instance
        model = cls(pretrained_model)        
        return model

class ScalarHead(nn.Module):
    r"""
    The ValueHead class implements a head for autoregressive that returns a scalar for each output token.
    The weights of the value head need to be in FP32.
    """

    def __init__(self, config, bias=True):
        super().__init__()
        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size
        # use a linear head to output a scalar
        self.summary =  nn.Linear(hidden_size, 1, bias=bias)    

    def forward(self, hidden_states):
        # Keep the scalar head weights in FP32 for stability, but accept BF16/FP16
        # backbone activations by casting only at the head boundary.
        head_dtype = self.summary.weight.dtype
        if hidden_states.dtype != head_dtype:
            hidden_states = hidden_states.to(head_dtype)
        output = self.summary(hidden_states)
        return output

class AutoModelForCausalLMWithScalarHead(PreTrainedModelWrapper):
    r"""
    A transformer backbone with a scalar head.
    Used for reward model / critic model.

    Important:
    We intentionally load `AutoModel` instead of `AutoModelForCausalLM` so the
    reward model does NOT materialize full vocabulary logits. For models like
    Gemma2, calling the CausalLM class computes `lm_head(hidden_states)` over the
    full vocab, which is unnecessary for scalar rewards and can cause massive OOM.
    """
    transformers_parent_class = AutoModel
    
    def __init__(self, pretrained_model):
        super().__init__(pretrained_model)
        self.scalar_head = ScalarHead(self.pretrained_model.config)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *args, **kwargs):
        r"""
        Instantiates a new model from a pretrained model from `transformers`.
        """
        # First, load the pre-trained model using the parent-class
        if isinstance(pretrained_model_path, str):
            pretrained_model = cls.transformers_parent_class.from_pretrained(
                pretrained_model_path, *args, **kwargs
            )
        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string, "
                f"but is {type(pretrained_model_path)}"
            )
        
        # call the __init__ method to create an instance
        model = cls(pretrained_model)
        
        # Try to load scalar_head weights from saved files if they exist
        if hasattr(model, 'scalar_head'):
            scalar_head_path = os.path.join(pretrained_model_path, "scalar_head.pt")
            scalar_head_quality_path = os.path.join(pretrained_model_path, "scalar_head_quality.pt")
            if os.path.exists(scalar_head_path):
                scalar_head_state_dict = torch.load(scalar_head_path, map_location='cpu')
                model.scalar_head.load_state_dict(scalar_head_state_dict)
            elif os.path.exists(scalar_head_quality_path):
                scalar_head_state_dict = torch.load(scalar_head_quality_path, map_location='cpu')
                model.scalar_head.load_state_dict(scalar_head_state_dict)
        
        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Run the backbone only and produce per-token scalar values.
        We explicitly avoid computing LM logits.
        """
        kwargs.pop("labels", None)
        kwargs.pop("output_hidden_states", None)
        kwargs["use_cache"] = False
        kwargs["return_dict"] = True
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden_state = base_model_output.last_hidden_state
        scalar_value = self.scalar_head(last_hidden_state).squeeze(-1)
        return scalar_value
    
    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        r"""
        Save the model to a directory so that it can be re-loaded using the
        `from_pretrained` method.
        
        Args:
            save_directory (str): Directory to save the model to
            state_dict (dict, optional): If provided, use this state_dict instead of the model's current state
            **kwargs: Additional arguments passed to the underlying model's save_pretrained method
        """
        if state_dict is None:
            # Normal path: let the base model save itself
            self.pretrained_model.save_pretrained(save_directory, **kwargs)
            scalar_sd = self.scalar_head.state_dict()
        else:
            # Merged state_dict path: extract base model weights and scalar head weights
            # Extract weights that start with "pretrained_model." for the base model
            base_sd = {k[len("pretrained_model."):]: v
                       for k, v in state_dict.items()
                       if k.startswith("pretrained_model.")}
            self.pretrained_model.save_pretrained(save_directory, state_dict=base_sd, **kwargs)
            
            # Extract weights that start with "scalar_head." for the scalar head
            scalar_sd = {k[len("scalar_head."):]: v
                         for k, v in state_dict.items()
                         if k.startswith("scalar_head.")}
        
        # Save the scalar_head weights
        scalar_head_path = os.path.join(save_directory, "scalar_head.pt")
        torch.save(scalar_sd, scalar_head_path)
        
        # Save the model config to indicate this is a wrapper model
        config = self.pretrained_model.config
        config.is_wrapper_model = True
        config.wrapper_type = "AutoModelForCausalLMWithScalarHead"
        config.save_pretrained(save_directory)

class AutoModelForCausalLMWithScalarHeadODIN(PreTrainedModelWrapper):
    r"""
    A transformer backbone with two scalar heads for ODIN reward shaping.
    Again, we load `AutoModel` rather than `AutoModelForCausalLM` to avoid the
    unnecessary full-vocabulary logits path.
    """
    transformers_parent_class = AutoModel
    
    def __init__(self, pretrained_model):
        super().__init__(pretrained_model)
        self.scalar_head_quality = ScalarHead(self.pretrained_model.config, bias=False)
        self.scalar_head_length = ScalarHead(self.pretrained_model.config, bias=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *args, **kwargs):
        r"""
        Instantiates a new model from a pretrained model from `transformers`.
        """
        # First, load the pre-trained model using the parent-class
        if isinstance(pretrained_model_path, str):
            pretrained_model = cls.transformers_parent_class.from_pretrained(
                pretrained_model_path, *args, **kwargs
            )
        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string, "
                f"but is {type(pretrained_model_path)}"
            )
        
        # call the __init__ method to create an instance
        model = cls(pretrained_model)
        
        # Try to load scalar_head weights from saved files if they exist
        if hasattr(model, 'scalar_head_quality') and hasattr(model, 'scalar_head_length'):
            scalar_head_quality_path = os.path.join(pretrained_model_path, "scalar_head_quality.pt")
            scalar_head_length_path = os.path.join(pretrained_model_path, "scalar_head_length.pt")
            if os.path.exists(scalar_head_quality_path):
                scalar_head_quality_state_dict = torch.load(scalar_head_quality_path, map_location='cpu')
                model.scalar_head_quality.load_state_dict(scalar_head_quality_state_dict)
            if os.path.exists(scalar_head_length_path):
                scalar_head_length_state_dict = torch.load(scalar_head_length_path, map_location='cpu')
                model.scalar_head_length.load_state_dict(scalar_head_length_state_dict)
        
        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Run the backbone only and produce per-token scalar values.
        We explicitly avoid computing LM logits.
        """
        kwargs.pop("labels", None)
        kwargs.pop("output_hidden_states", None)
        kwargs["use_cache"] = False
        kwargs["return_dict"] = True
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden_state = base_model_output.last_hidden_state
        scalar_value_quality = self._normalized_head_forward(self.scalar_head_quality, last_hidden_state).squeeze(-1)
        scalar_value_length = self._normalized_head_forward(self.scalar_head_length, last_hidden_state).squeeze(-1)
        return scalar_value_quality, scalar_value_length

    @staticmethod
    def _normalized_head_forward(head, hidden_states):
        weight = head.summary.weight
        if hidden_states.dtype != weight.dtype:
            hidden_states = hidden_states.to(weight.dtype)
        normalized_weight = torch.nn.functional.normalize(weight, p=2, dim=-1)
        return torch.nn.functional.linear(hidden_states, normalized_weight, head.summary.bias)
    
    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        r"""
        Save the model to a directory so that it can be re-loaded using the
        `from_pretrained` method.
        
        Args:
            save_directory (str): Directory to save the model to
            state_dict (dict, optional): If provided, use this state_dict instead of the model's current state
            **kwargs: Additional arguments passed to the underlying model's save_pretrained method
        """
        if state_dict is None:
            # Normal path: let the base model save itself
            self.pretrained_model.save_pretrained(save_directory, **kwargs)
            scalar_sd_quality = self.scalar_head_quality.state_dict()
            scalar_sd_length = self.scalar_head_length.state_dict()
        else:
            # Merged state_dict path: extract base model weights and scalar head weights
            # Extract weights that start with "pretrained_model." for the base model
            base_sd = {k[len("pretrained_model."):]: v
                       for k, v in state_dict.items()
                       if k.startswith("pretrained_model.")}
            self.pretrained_model.save_pretrained(save_directory, state_dict=base_sd, **kwargs)
            
            # Extract weights that start with "scalar_head_quality." for the quality head
            scalar_sd_quality = {k[len("scalar_head_quality."):]: v
                                for k, v in state_dict.items()
                                if k.startswith("scalar_head_quality.")}
            
            # Extract weights that start with "scalar_head_length." for the length head
            scalar_sd_length = {k[len("scalar_head_length."):]: v
                               for k, v in state_dict.items()
                               if k.startswith("scalar_head_length.")}
        
        # Save the scalar_head weights
        scalar_head_quality_path = os.path.join(save_directory, "scalar_head_quality.pt")
        scalar_head_length_path = os.path.join(save_directory, "scalar_head_length.pt")
        torch.save(scalar_sd_quality, scalar_head_quality_path)
        torch.save(scalar_sd_length, scalar_head_length_path)
        
        # Save the model config to indicate this is a wrapper model
        config = self.pretrained_model.config
        config.is_wrapper_model = True
        config.wrapper_type = "AutoModelForCausalLMWithScalarHeadODIN"
        config.save_pretrained(save_directory)

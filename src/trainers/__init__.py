from .a2c_trainer import A2CTrainer
from .grpo_trainer import GRPOTrainer
from .paired_trainer import DPOTrainer, RewardTrainer, RewardTrainerODIN
from .ppo_trainer import PPOTrainer
from .unpaired_trainer import GenrefsTrainer, SFTTrainer


TRAINER_REGISTRY = {
    "A2CTrainer": A2CTrainer,
    "DPOTrainer": DPOTrainer,
    "GenrefsTrainer": GenrefsTrainer,
    "GRPOTrainer": GRPOTrainer,
    "PPOTrainer": PPOTrainer,
    "RewardTrainer": RewardTrainer,
    "RewardTrainerODIN": RewardTrainerODIN,
    "SFTTrainer": SFTTrainer,
}


def get_trainer_class(name: str):
    try:
        return TRAINER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown trainer class: {name}") from exc

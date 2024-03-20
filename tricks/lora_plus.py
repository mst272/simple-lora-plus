from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn
from functools import reduce
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from peft.tuners import lora
from transformers import Trainer, HfArgumentParser
from transformers.utils import is_sagemaker_mp_enabled, logging

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

logger = logging.get_logger(__name__)


# LayerNorm层及bias等不需要进行weight decay
# ALL_LAYERNORM_LAYERS = [nn.LayerNorm, LlamaRMSNorm]

# 论文中的推荐相关参数设置
LORA_LR_RATIO = 16
LORA_LR_EMBEDDING = 1e-6
WEIGHT_DECAY = 0.0


def get_modules(name, model):
    """
    通过名字获取module
    """
    if "lora" in name:
        parent_idx = 2
    else:
        parent_idx = 1

    module_name = name.split(sep=".")[:-parent_idx]
    module = reduce(getattr, module_name, model)
    return module


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def create_lorap_optimizer(model, lora_lr_ratio, optimizer_cls, optimizer_kwargs, lora_lr_embedding=None):
    if lora_lr_embedding is None:
        lora_lr_embedding = 1e-6
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    parameters = {
        "A": {},
        "B": {},
        "B_no_decay": {},
        "embedding": {}
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = get_modules(name, model)
        if isinstance(module, lora.Embedding):
            parameters['embedding'][name] = param
        elif "lora_B" in name:
            if name in decay_parameters:
                parameters['B'][name] = param
            else:
                parameters['B_no_decay'][name] = param
        else:
            parameters['A'][name] = param

    apply_param_groups = ""
    for group in parameters:
        apply_param_groups += f"{group}\n {list(parameters[group].keys())}\n\n"
    logger.info(apply_param_groups)

    lr = optimizer_kwargs["lr"]
    weight_decay = WEIGHT_DECAY

    optimizer_grouped_parameters = [
        {
            "params": list(parameters["A"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(parameters["embedding"].values()),
            "weight_decay": weight_decay,
            "lr": lora_lr_embedding,
        },
        {
            "params": list(parameters["B"].values()),
            "weight_decay": weight_decay,
            "lr": lr * lora_lr_ratio,
        },
        {
            "params": list(parameters["B_no_decay"].values()),
            "weight_decay": 0.0,
            "lr": lr * lora_lr_ratio,
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    # transformers trainer
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum(
                    {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                )
                logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped / 2 ** 20}M params")

    return optimizer


# 重写Trainer 的 create_optimizer方法
class LoraPlusTrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            lora_lr_ratio = LORA_LR_RATIO
            lora_lr_embedding = LORA_LR_EMBEDDING

            self.optimizer = create_lorap_optimizer(opt_model, lora_lr_ratio, optimizer_cls, optimizer_kwargs,
                                                    lora_lr_embedding)
        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

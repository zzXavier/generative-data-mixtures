"""PEFT attention adapters used by the conditional image generator."""
import random
import numpy as np
import torch
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file


def add_adapter(unet,rank=4):
    unet.add_adapter(LoraConfig(r=rank,lora_alpha=rank,init_lora_weights="gaussian",
                                target_modules=["to_q","to_k","to_v","to_out.0"]))
    return [p for p in unet.parameters() if p.requires_grad]


def trainable_state(unet):
    return {k:v.detach().float().cpu().contiguous().clone() for k,v in get_peft_model_state_dict(unet).items()}


def save_reload(unet,initial,dest):
    state=trainable_state(unet)
    update=float(sum((v-initial[k]).square().sum().item() for k,v in state.items())**.5)
    if not np.isfinite(update) or update<=0:
        raise RuntimeError("No finite LoRA update")
    save_file(state,str(dest))
    for p in unet.parameters():
        if p.requires_grad:
            p.data.zero_()
    set_peft_model_state_dict(unet,load_file(str(dest)))
    recovered=trainable_state(unet)
    if any(not torch.equal(v,recovered[k]) for k,v in state.items()):
        raise RuntimeError("LoRA reload changed parameters")
    return update


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

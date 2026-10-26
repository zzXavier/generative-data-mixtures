"""Small conditional DDPM using Hugging Face Diffusers and PEFT attention LoRA."""
import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from torch.nn import functional as F

def new_model():
    return UNet2DConditionModel(
        sample_size=8,in_channels=1,out_channels=1,layers_per_block=1,
        block_out_channels=(12,24),down_block_types=("CrossAttnDownBlock2D","DownBlock2D"),
        up_block_types=("UpBlock2D","CrossAttnUpBlock2D"),norm_num_groups=3,
        attention_head_dim=(3,3),cross_attention_dim=12)


def context(digit, polarity):
    # Each task caption has exactly two tokens: true/perturbed digit and polarity.
    return torch.stack([F.one_hot(digit,12),F.one_hot(10+polarity,12)],dim=1).float()


def scheduler(cfg):
    return DDPMScheduler(num_train_timesteps=cfg["train_timesteps"],beta_schedule="squaredcos_cap_v2",
                         prediction_type="epsilon",clip_sample=True)


def train(model, pixels, labels, styles, cfg, steps, lr):
    model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.01)
    sched=scheduler(cfg)
    rng=torch.Generator().manual_seed(cfg["seed"]+11)
    logs=[]
    for step in range(steps):
        idx=torch.randint(len(pixels),(cfg["batch_size"],),generator=rng)
        clean=pixels[idx]
        eps=torch.randn(clean.shape,generator=rng)
        ts=torch.randint(sched.config.num_train_timesteps,(len(idx),),generator=rng)
        pred=model(sched.add_noise(clean,eps,ts),ts,encoder_hidden_states=context(labels[idx],styles[idx])).sample
        loss=F.mse_loss(pred,eps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        logs.append(float(loss.detach()))
    return logs


@torch.inference_mode()
def draw_samples(model, cfg):
    model.eval()
    marks=[(d,p,s) for p in range(2) for d in range(10) for s in range(cfg["generation_seeds"])]
    rng=torch.Generator().manual_seed(cfg["seed"]+29)
    noise=torch.randn((len(marks),1,8,8),generator=rng)
    # A recurrent noise tensor is shared by all arms, per prompt/seed index.
    tokens=context(torch.tensor([m[0] for m in marks]),torch.tensor([m[1] for m in marks]))
    sched=DDIMScheduler.from_config(scheduler(cfg).config)
    sched.set_timesteps(cfg["sampling_steps"])
    for ts in sched.timesteps:
        pred=model(noise,ts,encoder_hidden_states=tokens).sample
        noise=sched.step(pred,ts,noise).prev_sample
    return ((noise[:,0]+1)/2).clamp(0,1).numpy(),marks


@torch.inference_mode()
def heldout_denoising(model,x,y,ids,cfg):
    rng=torch.Generator().manual_seed(cfg["seed"]+23)
    labels=torch.from_numpy(y[ids]).long()
    pol=torch.arange(len(ids))%2
    pixels=torch.from_numpy(x[ids].copy())
    pixels=torch.where(pol[:,None,None].bool(),1-pixels,pixels)[:,None]*2-1
    times=torch.randint(cfg["train_timesteps"],(len(ids),),generator=rng)
    eps=torch.randn(pixels.shape,generator=rng)
    noisy=scheduler(cfg).add_noise(pixels,eps,times)
    errors=[]
    for a in range(0,len(ids),32):
        sl=slice(a,a+32)
        pred=model(noisy[sl],times[sl],encoder_hidden_states=context(labels[sl],pol[sl])).sample
        errors.extend(((pred-eps[sl])**2).mean((1,2,3)).tolist())
    return [dict(source=int(i),mse=float(e)) for i,e in zip(ids,errors)]

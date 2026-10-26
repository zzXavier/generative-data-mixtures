import argparse
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME",str(ROOT/".cache"/"huggingface"))
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT","2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")


def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def reproduce(out,cfg):
    import numpy as np
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file,save_file
    from src.curation import ARMS,image_for,load,select
    from src.diffusion import new_model,train,draw_samples,heldout_denoising
    from src.report import score,evaluate
    from src.sd import seed_all,add_adapter,trainable_state,save_reload
    torch.set_num_threads(2)
    seed_all(cfg["seed"])
    x,y,splits,rows,judge=load(cfg["seed"])
    out.mkdir(parents=True,exist_ok=True)
    weights=out/"weights"
    weights.mkdir(exist_ok=True)
    save_json(out/"split.json",{s:v.tolist() for s,v in splits.items()})
    save_json(out/"config.json",cfg)
    base=new_model()
    ids=splits["base"]
    pol=torch.arange(len(ids))%2
    raw=torch.from_numpy(x[ids].copy())
    pixels=torch.where(pol[:,None,None].bool(),1-raw,raw)[:,None]*2-1
    since=time.monotonic()
    base_losses=train(base,pixels,torch.from_numpy(y[ids]).long(),pol,cfg,cfg["base_updates"],cfg["base_lr"])
    save_file({k:v.detach().contiguous() for k,v in base.state_dict().items()},str(weights/"base.safetensors"))
    logs={"base":{"loss":base_losses,"seconds":time.monotonic()-since}}
    judgetrue=float((judge.predict(x[splits["test"]].reshape(-1,64))==y[splits["test"]]).mean())
    selections, predictions, samples={},{},{}
    for arm in ("base",*ARMS):
        print(f"Running {arm}",flush=True)
        if arm!="base":
            since=time.monotonic()
            kept=select(rows,arm,cfg["selected_per_arm"],cfg["seed"],cfg["top_score_fraction"])
            selections[arm]=kept
            clean=torch.from_numpy(np.stack([image_for(x,r) for r in kept]))[:,None]*2-1
            digits=torch.tensor([r["digit"] for r in kept])
            styles=torch.tensor([r["polarity"] for r in kept])
            seed_all(cfg["seed"]+17)
            adapted=new_model()
            adapted.load_state_dict(load_file(str(weights/"base.safetensors")))
            adapted.requires_grad_(False)
            params=add_adapter(adapted,rank=4)
            initial=trainable_state(adapted)
            losses=train(adapted,clean,digits,styles,cfg,cfg["adapter_updates"],cfg["adapter_lr"])
            update=save_reload(adapted,initial,weights/f"{arm}.safetensors")
            logs[arm]=dict(loss=losses,seconds=time.monotonic()-since,lora_update_l2=update,
                           trainable_parameters=sum(p.numel() for p in params))
        # The model used for evaluation is rebuilt from disk for every arm.
        model=new_model()
        model.load_state_dict(load_file(str(weights/"base.safetensors")))
        if arm!="base":
            add_adapter(model,rank=4)
            set_peft_model_state_dict(model,load_file(str(weights/f"{arm}.safetensors")))
        model.eval()
        generated,marks=draw_samples(model,cfg)
        samples[arm]=generated
        predictions[arm]=dict(samples=score(generated,marks,judge),
                              heldout=heldout_denoising(model,x,y,splits["test"],cfg))
    save_json(out/"selection.json",selections);save_json(out/"predictions.json",predictions)
    save_json(out/"training.json",{"heldout_classifier_accuracy_on_real_images":judgetrue,"runs":logs})
    np.savez_compressed(out/"samples.npz",**samples)
    evaluate(out,out)


def main():
    parser=argparse.ArgumentParser(description="Fixed-budget curation study on rendered UCI digit captions")
    parser.add_argument("command",choices=("demo","reproduce","evaluate"))
    parser.add_argument("--out",type=Path,default=ROOT/"work")
    parser.add_argument("--run",type=Path,default=ROOT/"results")
    args=parser.parse_args()
    cfg=json.loads((ROOT/"configs/experiment.json").read_text(encoding="utf-8"))
    if args.command=="reproduce":
        reproduce(args.out,cfg)
    elif args.command=="evaluate":
        from src.report import evaluate
        evaluate(args.run,args.out)
    else:
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        from src.diffusion import new_model,draw_samples
        from src.sd import add_adapter
        from PIL import Image
        import numpy as np
        torch.set_num_threads(2)
        cfg=json.loads((args.run/"config.json").read_text(encoding="utf-8"))
        model=new_model()
        model.load_state_dict(load_file(str(args.run/"weights/base.safetensors")))
        add_adapter(model)
        set_peft_model_state_dict(model,load_file(str(args.run/"weights/combined-50.safetensors")))
        images,marks=draw_samples(model,cfg)
        imgs=[a for a,m in zip(images,marks) if m[1:]==(0,0)]
        args.out.mkdir(parents=True,exist_ok=True)
        picture=Image.fromarray((np.concatenate(imgs,axis=1)*255).round().astype("uint8"))
        picture.resize((640,64),Image.Resampling.NEAREST).save(args.out/"demo.png")
    print(args.out.resolve())


if __name__=="__main__":
    main()

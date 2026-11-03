import csv
import json
from pathlib import Path
from statistics import mean
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def score(images,marks,judge):
    rows=[]
    for image,(digit,polarity,seed) in zip(images,marks):
        border=np.concatenate([image[0],image[-1],image[1:-1,0],image[1:-1,-1]])
        pred_pol=int(border.mean()>.5)
        upright=1-image if pred_pol else image
        digit_pred=int(judge.predict(upright.reshape(1,-1))[0])
        rows.append(dict(digit=digit,polarity=polarity,seed_index=seed,predicted_digit=digit_pred,
                         predicted_polarity=pred_pol,border_luminance=float(border.mean())))
    return rows


def evaluate(run,out):
    out.mkdir(parents=True,exist_ok=True)
    records=json.loads((run/"predictions.json").read_text(encoding="utf-8"))
    selection=json.loads((run/"selection.json").read_text(encoding="utf-8"))
    samples=np.load(run/"samples.npz")
    table=[]
    for arm,values in records.items():
        rows=values["samples"]
        keep=selection.get(arm,[])
        table.append(dict(arm=arm,images=len(keep),generated=len(rows),
                          unique_groups=len({r["signature"] for r in keep}),
                          incorrect_captions=sum(r["digit"]!=r["true_digit"] for r in keep),
                          digit_accuracy=mean(int(r["digit"]==r["predicted_digit"]) for r in rows),
                          polarity_accuracy=mean(int(r["polarity"]==r["predicted_polarity"]) for r in rows),
                          joint_accuracy=mean(int(r["polarity"]==r["predicted_polarity"] and r["digit"]==r["predicted_digit"]) for r in rows),
                          heldout_denoising_mse=mean(r["mse"] for r in values["heldout"])))
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=table[0].keys())
        writer.writeheader();writer.writerows(table)
    draw_sheet(records,samples,out/"comparison.png")
    return table


def draw_sheet(records,images,out):
    width=800
    step=64
    canvas=Image.new("RGB",(width,70+len(records)*step),"#f2f4f6")
    d=ImageDraw.Draw(canvas)
    f=ImageFont.load_default(size=16)
    d.text((12,8),"Same seed and requested digit, sampled from each model",font=f,fill="#172936")
    for i in range(10):
        d.text((171+i*60,40),str(i),font=f,fill="#172936")
    for n,(arm,value) in enumerate(records.items()):
        y=65+n*step
        d.text((12,y+13),arm,font=f,fill="#172936")
        for i in range(10):
            idx=next(j for j,r in enumerate(value["samples"]) if r["digit"]==i and r["polarity"]==0 and r["seed_index"]==0)
            img=Image.fromarray((images[arm][idx]*255).round().astype("uint8")).resize((50,50),Image.Resampling.NEAREST)
            canvas.paste(img,(160+i*60,y))
    canvas.save(out)

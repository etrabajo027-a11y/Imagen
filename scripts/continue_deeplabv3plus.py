import json, os, random, re, time, resource
from pathlib import Path, PurePosixPath
from io import BytesIO

import numpy as np
from PIL import Image
from remotezip import RemoteZip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50

SEED=42
TRAIN_N=int(os.getenv("TRAIN_SAMPLES","800"))
VAL_N=int(os.getenv("VAL_SAMPLES","500"))
EPOCHS=int(os.getenv("EPOCHS","5"))
W,H=512,288
CLASSES=19
BASE="http://128.32.162.150/bdd100k/"

def stem(p): return PurePosixPath(p).stem
def mask_stem(p): return re.sub(r"_train_id$","",stem(p),flags=re.I)

def prepare():
    for split in ("train","val"):
        (Path("training-data")/split/"images").mkdir(parents=True,exist_ok=True)
        (Path("training-data")/split/"masks").mkdir(parents=True,exist_ok=True)
    start=time.time()
    with RemoteZip(BASE+"bdd100k_images_10k.zip") as iz, RemoteZip(BASE+"bdd100k_seg_maps.zip") as mz:
        ins, mns=iz.namelist(),mz.namelist()
        ti=[p for p in ins if p.lower().endswith((".jpg",".jpeg")) and "/train/" in ("/"+p.lower())]
        vi=[p for p in ins if p.lower().endswith((".jpg",".jpeg")) and "/val/" in ("/"+p.lower())]
        tm=[p for p in mns if p.lower().endswith("_train_id.png") and "/train/" in ("/"+p.lower())]
        vm=[p for p in mns if p.lower().endswith("_train_id.png") and "/val/" in ("/"+p.lower())]
        tmap={mask_stem(p):p for p in tm}; vmap={mask_stem(p):p for p in vm}
        tp=[(p,tmap[stem(p)]) for p in ti if stem(p) in tmap]
        vp=[(p,vmap[stem(p)]) for p in vi if stem(p) in vmap]
        if len(tp)!=7000 or len(vp)!=1000: raise RuntimeError(f"BDD100K pair counts unexpected: {len(tp)}, {len(vp)}")
        rng=random.Random(SEED); rng.shuffle(tp); rng.shuffle(vp)
        for split,pairs in (("train",tp[:TRAIN_N]),("val",vp[:VAL_N])):
            for n,(ip,mp) in enumerate(pairs,1):
                ipath=Path("training-data")/split/"images"/(stem(ip)+".jpg")
                mpath=Path("training-data")/split/"masks"/(stem(ip)+".png")
                with Image.open(BytesIO(iz.read(ip))) as im:
                    im.convert("RGB").resize((W,H),Image.Resampling.BILINEAR).save(ipath,quality=95)
                with Image.open(BytesIO(mz.read(mp))) as m:
                    a=np.asarray(m)
                    ids=set(int(x) for x in np.unique(a))
                    if not ids.issubset(set(range(CLASSES))|{255}): raise RuntimeError(f"Invalid IDs: {ids}")
                    m.resize((W,H),Image.Resampling.NEAREST).save(mpath)
                if n%100==0: print(f"Prepared {split}: {n}/{len(pairs)}",flush=True)
    report={"status":"PASS","available_train_pairs":len(tp),"available_val_pairs":len(vp),
            "train_samples":TRAIN_N,"validation_samples":VAL_N,"resolution":[W,H],
            "seed":SEED,"seconds":round(time.time()-start,2)}
    Path("training-results").mkdir(exist_ok=True)
    Path("training-results/data_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

class ASPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches=nn.ModuleList([
            nn.Sequential(nn.Conv2d(2048,256,1,bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True)),
            *[nn.Sequential(nn.Conv2d(2048,256,3,padding=d,dilation=d,bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True)) for d in (6,12,18)]])
        self.pool=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(2048,256,1,bias=False),nn.ReLU(inplace=True))
        self.project=nn.Sequential(nn.Conv2d(1280,256,1,bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True),nn.Dropout(.1))
    def forward(self,x):
        p=F.interpolate(self.pool(x),size=x.shape[-2:],mode="bilinear",align_corners=False)
        return self.project(torch.cat([*(b(x) for b in self.branches),p],1))

class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super().__init__()
        b=resnet50(weights=None,replace_stride_with_dilation=[False,False,True])
        self.stem=nn.Sequential(b.conv1,b.bn1,b.relu,b.maxpool)
        self.layer1,self.layer2,self.layer3,self.layer4=b.layer1,b.layer2,b.layer3,b.layer4
        self.aspp=ASPP()
        self.low=nn.Sequential(nn.Conv2d(256,48,1,bias=False),nn.BatchNorm2d(48),nn.ReLU(inplace=True))
        self.decoder=nn.Sequential(nn.Conv2d(304,256,3,padding=1,bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1,bias=False),nn.BatchNorm2d(256),nn.ReLU(inplace=True),nn.Conv2d(256,19,1))
    def forward(self,x):
        size=x.shape[-2:]; x=self.stem(x); low=self.layer1(x)
        x=self.layer4(self.layer3(self.layer2(low))); x=self.aspp(x)
        x=F.interpolate(x,size=low.shape[-2:],mode="bilinear",align_corners=False)
        x=self.decoder(torch.cat([x,self.low(low)],1))
        return F.interpolate(x,size=size,mode="bilinear",align_corners=False)

class BDD(Dataset):
    def __init__(self,split):
        self.images=sorted((Path("training-data")/split/"images").glob("*.jpg"))
        self.maskdir=Path("training-data")/split/"masks"
        if not self.images: raise RuntimeError(f"No images for {split}")
    def __len__(self): return len(self.images)
    def __getitem__(self,i):
        p=self.images[i]
        a=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.
        a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        m=np.asarray(Image.open(self.maskdir/(p.stem+".png")),dtype=np.int64).copy()
        return torch.from_numpy(a.transpose(2,0,1)),torch.from_numpy(m)

def class_weights(dataset):
    counts=np.zeros(CLASSES,dtype=np.int64)
    for i in range(len(dataset)):
        _,m=dataset[i]
        vals,c=np.unique(m.numpy(),return_counts=True)
        for v,n in zip(vals,c):
            if 0<=v<CLASSES: counts[v]+=n
    present=counts>0
    freq=counts[present]/counts[present].sum()
    w=np.ones(CLASSES,dtype=np.float32)
    # Moderate inverse-square-root weighting; cap to avoid unstable gradients.
    w[present]=np.clip(1/np.sqrt(freq),1,5)
    w[present]/=w[present].mean()
    return torch.tensor(w),counts.tolist()

def evaluate(model,loader,criterion):
    model.eval(); losses=[]; cm=torch.zeros((CLASSES,CLASSES),dtype=torch.int64)
    with torch.inference_mode():
        for step,(x,y) in enumerate(loader,1):
            z=model(x); losses.append(float(criterion(z,y)))
            p=z.argmax(1).reshape(-1); t=y.reshape(-1); v=t!=255
            cm+=torch.bincount(CLASSES*t[v]+p[v],minlength=CLASSES**2).reshape(CLASSES,CLASSES)
            if step%100==0 or step==len(loader): print(f"Validation {step}/{len(loader)}",flush=True)
    inter=cm.diag().double(); union=cm.sum(0).double()+cm.sum(1).double()-inter
    present=union>0; iou=inter/union.clamp_min(1)
    return float(np.mean(losses)),float(iou[present].mean()),float(iou.mean()),float(inter.sum()/cm.sum().clamp_min(1)),iou.tolist(),cm.tolist()

def main():
    torch.set_num_threads(2); torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    prepare()
    train_ds, val_ds=BDD("train"),BDD("val")
    train_loader=DataLoader(train_ds,batch_size=1,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0)
    model=DeepLabV3Plus()
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-5)
    checkpoint=torch.load("training-results/deeplabv3plus_last.pth",map_location="cpu",weights_only=False)
    if checkpoint.get("num_classes")!=19 or checkpoint.get("input_size")!=[W,H]:
        raise RuntimeError("Checkpoint architecture metadata mismatch")
    model.load_state_dict(checkpoint["model_state_dict"],strict=True)
    if "optimizer_state_dict" not in checkpoint: raise RuntimeError("Optimizer state missing; refusing fresh restart")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # Lower LR after restoring optimizer to make continuation conservative.
    for group in optimizer.param_groups: group["lr"]=3e-5
    weights,counts=class_weights(train_ds)
    criterion=nn.CrossEntropyLoss(weight=weights,ignore_index=255)
    start_epoch=int(checkpoint["epoch"]); best=float(checkpoint.get("metrics",{}).get("miou",0))
    best_path=Path("training-results/deeplabv3plus_best.pth")
    history_path=Path("training-results/continuation_history.json")
    history=[]
    started=time.time()
    print(f"RESUME from epoch={start_epoch}, train={len(train_ds)}, val={len(val_ds)}, extra_epochs={EPOCHS}, lr=3e-5",flush=True)
    print(f"Class pixel counts: {counts}",flush=True)
    for e in range(1,EPOCHS+1):
        epoch_start=time.time(); model.train(); losses=[]
        for step,(x,y) in enumerate(train_loader,1):
            optimizer.zero_grad(set_to_none=True); logits=model(x); loss=criterion(logits,y)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            if step%50==0 or step==len(train_loader):
                print(f"Epoch {start_epoch+e} train {step}/{len(train_loader)} loss={losses[-1]:.4f}",flush=True)
        vl,miou,miou19,acc,ious,cm=evaluate(model,val_loader,criterion)
        row={"epoch":start_epoch+e,"train_loss":float(np.mean(losses)),"val_loss":vl,
             "miou_present":miou,"miou_all19":miou19,"pixel_accuracy":acc,"class_iou":ious,
             "epoch_seconds":round(time.time()-epoch_start,2)}
        history.append(row)
        state={"epoch":start_epoch+e,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
               "metrics":{"miou":miou,"train_loss":row["train_loss"],"val_loss":vl,"pixel_accuracy":acc},
               "num_classes":19,"input_size":[W,H]}
        torch.save(state,"training-results/deeplabv3plus_last.pth")
        if miou>best:
            best=miou; torch.save(state,best_path)
            print(f"New best checkpoint: mIoU={best:.5f}",flush=True)
        history_path.write_text(json.dumps(history,indent=2))
        print("EPOCH SUMMARY "+json.dumps(row),flush=True)
    report={"status":"PASS","mode":"resume_from_existing_checkpoint","start_epoch":start_epoch,
            "epochs_completed":start_epoch+EPOCHS,"additional_epochs":EPOCHS,"train_samples":TRAIN_N,
            "validation_samples":VAL_N,"best_miou_present":best,"class_weights":weights.tolist(),
            "class_pixel_counts":counts,"learning_rate":3e-5,"total_seconds":round(time.time()-started,2),
            "max_rss_mb":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1)}
    Path("training-results/continuation_report.json").write_text(json.dumps(report,indent=2))
    print("CONTINUATION FINAL REPORT\n"+json.dumps(report,indent=2),flush=True)

if __name__=="__main__": main()

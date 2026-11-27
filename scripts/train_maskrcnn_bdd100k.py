#!/usr/bin/env python3
"""Small, real-data BDD100K instance segmentation training on CPU.

Reads selected JPEG/PNG members via HTTP range requests; does not download the
full image archive. Uses official 16-bit instance IDs: (B << 8) | A.
"""
import json, os, random, re, time
from pathlib import Path, PurePosixPath
from io import BytesIO
from urllib.parse import urljoin
import requests
import numpy as np
from PIL import Image
from remotezip import RemoteZip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

SEED=42
TRAIN_N=int(os.getenv("TRAIN_SAMPLES","100"))
VAL_N=int(os.getenv("VAL_SAMPLES","30"))
EPOCHS=int(os.getenv("EPOCHS","2"))
W,H=512,288
BASE="http://128.32.162.150/bdd100k/"
CATEGORIES={1:"pedestrian",2:"rider",3:"car",4:"truck",5:"bus",6:"train",7:"motorcycle",8:"bicycle"}
OUT=Path("maskrcnn-results"); OUT.mkdir(exist_ok=True)

def stem(p): return PurePosixPath(p).stem

def discover_instance_archive():
    # Current BDD100K documentation points to ETH Zurich's data mirror.
    # Keep the historical S3 release URL as a fallback, since mirrors may differ.
    urls=[
      "https://dl.cv.ethz.ch/bdd100k/data/bdd100k_ins_seg_labels_trainval.zip",
      "https://bdd-data-storage-release.s3.us-west-2.amazonaws.com/bdd100k/2021/bdd100k_ins_seg_labels_trainval.zip",
    ]
    errors=[]
    for url in urls:
        name=url.rsplit("/",1)[-1]
        try:
            with RemoteZip(url) as z:
                files=z.namelist()
                tr=[p for p in files if p.lower().endswith(".png") and "bitmasks" in p.lower() and "/train/" in ("/"+p.lower())]
                va=[p for p in files if p.lower().endswith(".png") and "bitmasks" in p.lower() and "/val/" in ("/"+p.lower())]
                if tr and va:
                    print(f"Using official instance archive: {name}; train_masks={len(tr)}, val_masks={len(va)}",flush=True)
                    return url,name,tr,va,urls
                errors.append(f"{name}: ZIP opened but expected bitmasks/train and bitmasks/val paths were not found")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise RuntimeError("Could not read official BDD100K instance-mask archive: "+"; ".join(errors))

def decode_rgba(raw):
    a=np.asarray(raw.convert("RGBA"),dtype=np.uint8)
    cat=a[...,0]; flags=a[...,1]
    instance=(a[...,2].astype(np.uint16)<<8)|a[...,3].astype(np.uint16)
    valid=(cat>0)&((flags&1)==0)&((flags&2)==0)&(instance>0)
    masks=[]; labels=[]
    if valid.any():
        for cid,iid in np.unique(np.stack([cat[valid],instance[valid]],axis=1),axis=0):
            cid=int(cid); iid=int(iid)
            if cid not in CATEGORIES: continue
            m=valid&(cat==cid)&(instance==iid)
            if int(m.sum())<4: continue
            resized=np.asarray(Image.fromarray(m.astype(np.uint8)*255).resize((W,H),Image.Resampling.NEAREST))>0
            if int(resized.sum())>=4: masks.append(resized.astype(np.uint8)); labels.append(cid)
    if not masks: return np.zeros((0,H,W),np.uint8),np.zeros((0,),np.int64)
    return np.stack(masks),np.asarray(labels,dtype=np.int64)

def prepare():
    started=time.time()
    label_url,label_name,train_masks,val_masks,listed=discover_instance_archive()
    image_url=BASE+"bdd100k_images_10k.zip"
    with RemoteZip(image_url) as iz, RemoteZip(label_url) as lz:
        image_files=iz.namelist()
        tr_img=[p for p in image_files if p.lower().endswith((".jpg",".jpeg")) and "/train/" in ("/"+p.lower())]
        va_img=[p for p in image_files if p.lower().endswith((".jpg",".jpeg")) and "/val/" in ("/"+p.lower())]
        tm={stem(p):p for p in train_masks}; vm={stem(p):p for p in val_masks}
        ti={stem(p):p for p in tr_img}; vi={stem(p):p for p in va_img}
        tp=sorted(set(ti)&set(tm)); vp=sorted(set(vi)&set(vm))
        if len(tp)<TRAIN_N or len(vp)<VAL_N:
            raise RuntimeError(f"Insufficient real image-mask pairs: train={len(tp)}, val={len(vp)}")
        rng=random.Random(SEED); rng.shuffle(tp); rng.shuffle(vp)
        selected={"train":tp[:TRAIN_N],"val":vp[:VAL_N]}
        report={"label_archive":label_name,"label_archive_url":label_url,
          "directory_zip_links":listed,"available_train_pairs":len(set(ti)&set(tm)),
          "available_val_pairs":len(set(vi)&set(vm)),"train_samples_requested":TRAIN_N,
          "val_samples_requested":VAL_N,"resolution":[W,H],"seed":SEED,"splits":{}}
        for split,ids in selected.items():
            (OUT/split).mkdir(exist_ok=True); rows=[]
            imap=ti if split=="train" else vi; mmap=tm if split=="train" else vm
            for n,key in enumerate(ids,1):
                with Image.open(BytesIO(iz.read(imap[key]))) as im:
                    image=im.convert("RGB").resize((W,H),Image.Resampling.BILINEAR)
                with Image.open(BytesIO(lz.read(mmap[key]))) as mask:
                    masks,labels=decode_rgba(mask)
                np.savez_compressed(OUT/split/(key+".npz"),masks=masks,labels=labels)
                image.save(OUT/split/(key+".jpg"),quality=90)
                rows.append({"image":key,"instances":int(len(labels))})
                if n%25==0 or n==len(ids): print(f"Prepared {split}: {n}/{len(ids)}",flush=True)
            report["splits"][split]={"images":len(rows),"images_with_instances":sum(x["instances"]>0 for x in rows),
              "instances":sum(x["instances"] for x in rows)}
    report["prepare_seconds"]=round(time.time()-started,2)
    (OUT/"data_report.json").write_text(json.dumps(report,indent=2))
    print("DATA REPORT\n"+json.dumps(report,indent=2),flush=True)

class BDDInstance(Dataset):
    def __init__(self,split): self.paths=sorted((OUT/split).glob("*.jpg"))
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        p=self.paths[i]
        im=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.
        image=torch.from_numpy(im.transpose(2,0,1).copy())
        z=np.load(p.with_suffix(".npz")); masks=z["masks"].astype(np.uint8); labels=z["labels"].astype(np.int64)
        if len(masks):
            boxes=[]
            for m in masks:
                y,x=np.where(m>0); boxes.append([x.min(),y.min(),x.max()+1,y.max()+1])
            boxes=torch.as_tensor(boxes,dtype=torch.float32)
            mt=torch.as_tensor(masks,dtype=torch.uint8); lt=torch.as_tensor(labels,dtype=torch.int64)
            area=mt.flatten(1).sum(1).float()
        else:
            boxes=torch.zeros((0,4)); mt=torch.zeros((0,H,W),dtype=torch.uint8)
            lt=torch.zeros((0,),dtype=torch.int64); area=torch.zeros((0,))
        target={"boxes":boxes,"labels":lt,"masks":mt,"image_id":torch.tensor([i]),
          "area":area,"iscrowd":torch.zeros((len(lt),),dtype=torch.int64)}
        return image,target

def collate(batch): return tuple(zip(*batch))

def build_model():
    model=maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
      min_size=H,max_size=W,box_detections_per_img=50)
    n=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(n,9)
    n=model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden=model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor=MaskRCNNPredictor(n,hidden,9)
    return model

def mask_iou(a,b):
    u=np.logical_or(a,b).sum()
    return float(np.logical_and(a,b).sum()/u) if u else 0.

def ap_for_class(records,cid,threshold):
    total=sum(int((t["labels"].numpy()==cid).sum()) for _,t in records)
    if total==0: return None
    preds=[]
    for idx,(pred,target) in enumerate(records):
        labels=pred["labels"].cpu().numpy(); scores=pred["scores"].cpu().numpy()
        pm=pred["masks"].cpu().numpy()[:,0]>.5
        for j in np.where(labels==cid)[0]: preds.append((float(scores[j]),idx,pm[j]))
    preds.sort(key=lambda x:x[0],reverse=True)
    matched={i:set() for i in range(len(records))}; tp=[]; fp=[]
    for score,idx,pmask in preds:
        t=records[idx][1]; gm=t["masks"].numpy()[t["labels"].numpy()==cid]>0
        best=0.; bestj=-1
        for j,g in enumerate(gm):
            if j in matched[idx]: continue
            v=mask_iou(pmask,g)
            if v>best: best,bestj=v,j
        if best>=threshold and bestj>=0: matched[idx].add(bestj); tp.append(1); fp.append(0)
        else: tp.append(0); fp.append(1)
    if not tp: return 0.
    tp=np.cumsum(tp); fp=np.cumsum(fp); recall=tp/total; precision=tp/np.maximum(tp+fp,1)
    return float(np.mean([precision[recall>=r].max() if np.any(recall>=r) else 0.
      for r in np.linspace(0,1,101)]))

def evaluate(model,loader):
    model.eval(); records=[]
    with torch.inference_mode():
        for n,(images,targets) in enumerate(loader,1):
            outputs=model(list(images)); records.extend(zip(outputs,targets))
            if n%10==0 or n==len(loader): print(f"Validation: {n}/{len(loader)}",flush=True)
    thresholds=[.50,.55,.60,.65,.70,.75,.80,.85,.90,.95]; by={}
    for th in thresholds:
        vals=[ap_for_class(records,c,th) for c in CATEGORIES]
        present=[v for v in vals if v is not None]
        by[f"{th:.2f}"]={"mAP":float(np.mean(present)) if present else 0.,
          "per_class":{CATEGORIES[c]:vals[c-1] for c in CATEGORIES}}
    return {"AP50":by["0.50"]["mAP"],"mAP50_95":float(np.mean([v["mAP"] for v in by.values()])),
      "by_iou":by,"validation_images":len(records)}

def main():
    torch.set_num_threads(2); torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    prepare(); train=BDDInstance("train"); val=BDDInstance("val")
    tl=DataLoader(train,batch_size=1,shuffle=True,num_workers=0,collate_fn=collate)
    vl=DataLoader(val,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate)
    model=build_model(); params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.SGD(params,lr=.001,momentum=.9,weight_decay=.0005)
    history=[]; best=-1.; started=time.time()
    for epoch in range(1,EPOCHS+1):
        model.train(); losses=[]; ep=time.time()
        for step,(images,targets) in enumerate(tl,1):
            ld=model(list(images),list(targets)); loss=sum(ld.values())
            if not torch.isfinite(loss): raise RuntimeError(f"Nonfinite losses: {ld}")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5.); opt.step(); losses.append(float(loss.detach()))
            if step%10==0 or step==len(tl):
                print(f"Epoch {epoch}/{EPOCHS} train {step}/{len(tl)} loss={losses[-1]:.4f}",flush=True)
        metrics=evaluate(model,vl)
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),"epoch_seconds":round(time.time()-ep,2),**metrics}
        history.append(row)
        state={"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),
          "categories":CATEGORIES,"input_size":[W,H],"metrics":metrics}
        torch.save(state,OUT/"maskrcnn_last.pth")
        if metrics["mAP50_95"]>best: best=metrics["mAP50_95"]; torch.save(state,OUT/"maskrcnn_best.pth")
        (OUT/"history.json").write_text(json.dumps(history,indent=2))
        print("EPOCH SUMMARY "+json.dumps(row),flush=True)
    report={"status":"PASS","model":"Mask R-CNN ResNet50-FPN (COCO pretrained; BDD100K 8 classes)",
      "train_samples":len(train),"validation_samples":len(val),"epochs":EPOCHS,
      "best_mAP50_95":best,"total_seconds":round(time.time()-started,2)}
    (OUT/"training_report.json").write_text(json.dumps(report,indent=2))
    print("TRAINING FINAL REPORT\n"+json.dumps(report,indent=2),flush=True)
if __name__=="__main__": main()

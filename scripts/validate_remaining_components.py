#!/usr/bin/env python3
"""Synthetic readiness tests for BDD100K instance masks and Mask R-CNN."""
import json
from pathlib import Path
import numpy as np
import torch

CATEGORIES={1:"pedestrian",2:"rider",3:"car",4:"truck",5:"bus",6:"train",7:"motorcycle",8:"bicycle"}

def decode_bdd_instance_rgba(rgba):
    arr=np.asarray(rgba,dtype=np.uint8)
    if arr.ndim!=3 or arr.shape[2]!=4: raise ValueError(f"Expected RGBA mask, got {arr.shape}")
    category=arr[...,0]; flags=arr[...,1]
    # Official BDD100K encodes ann_id as a 16-bit integer in B (high byte), A (low byte).
    instance=(arr[...,2].astype(np.uint16)<<8)|arr[...,3].astype(np.uint16)
    valid=(category>0)&((flags&1)==0)&((flags&2)==0)&(instance>0)
    instances=[]
    pairs=np.unique(np.stack([category[valid],instance[valid]],axis=-1),axis=0) if valid.any() else []
    for cat_id,inst_id in pairs:
        cat_id=int(cat_id); inst_id=int(inst_id)
        if cat_id not in CATEGORIES: continue
        mask=valid&(category==cat_id)&(instance==inst_id)
        if not mask.any(): continue
        ys,xs=np.where(mask)
        instances.append({"category_id":cat_id,"category":CATEGORIES[cat_id],"instance_id":inst_id,
          "mask":mask,"bbox_xyxy":[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1],
          "area":int(mask.sum())})
    return instances

def test_instance_decoder():
    rgba=np.zeros((32,48,4),dtype=np.uint8)
    rgba[3:15,4:18,0]=3; rgba[3:15,4:18,2]=1; rgba[3:15,4:18,3]=7
    rgba[18:29,25:42,0]=1; rgba[18:29,25:42,2]=1; rgba[18:29,25:42,3]=8
    rgba[0:2,0:2,0]=3; rgba[0:2,0:2,1]=1; rgba[0:2,0:2,3]=9
    instances=decode_bdd_instance_rgba(rgba)
    assert len(instances)==2,len(instances)
    assert {x["category"] for x in instances}=={"car","pedestrian"}
    assert {x["instance_id"] for x in instances}=={263,264}
    assert all(x["mask"].dtype==np.bool_ and x["area"]>0 for x in instances)
    return {"status":"PASS","instances_decoded":len(instances),"instance_ids":[263,264],
            "categories":["car","pedestrian"],"verified_16bit_BA_encoding":True}

def test_mask_rcnn():
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    torch.set_num_threads(2); torch.manual_seed(42)
    model=maskrcnn_resnet50_fpn(weights=None,weights_backbone=None,min_size=128,max_size=160,box_detections_per_img=10)
    n=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(n,9)
    n=model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden=model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor=MaskRCNNPredictor(n,hidden,9)
    image=torch.rand(3,128,160); masks=torch.zeros((2,128,160),dtype=torch.uint8)
    masks[0,20:60,20:70]=1; masks[1,65:110,85:135]=1
    target={"boxes":torch.tensor([[20.,20.,70.,60.],[85.,65.,135.,110.]]),
      "labels":torch.tensor([3,1]),"masks":masks,"image_id":torch.tensor([1]),
      "area":torch.tensor([2000.,2250.]),"iscrowd":torch.zeros(2,dtype=torch.int64)}
    model.train(); losses=model([image],[target]); total=sum(losses.values())
    assert torch.isfinite(total); total.backward()
    vals={k:float(v.detach()) for k,v in losses.items()}
    model.eval()
    with torch.inference_mode(): pred=model([image])
    assert set(("boxes","labels","scores","masks")).issubset(pred[0])
    return {"status":"PASS","num_classes_including_background":9,"losses":vals,
            "prediction_fields":sorted(pred[0].keys()),"predicted_instances":int(len(pred[0]["boxes"]))}

def test_environment_join():
    names={"a.jpg","b.jpg","c.jpg","unmatched.jpg"}
    labels=[{"name":"a.jpg","attributes":{"weather":"rainy","timeofday":"night","scene":"city street"}},
      {"name":"b.jpg","attributes":{"weather":"clear","timeofday":"daytime","scene":"highway"}},
      {"name":"c.jpg","attributes":{"weather":"foggy","timeofday":"dawn/dusk","scene":"residential"}}]
    by_name={Path(x["name"]).name:x.get("attributes",{}) for x in labels}
    rows=[{"image":n,**{k:by_name.get(n,{}).get(k,"unknown") for k in ("weather","timeofday","scene")}} for n in sorted(names)]
    matched=sum(x["weather"]!="unknown" for x in rows)
    assert matched==3 and next(x for x in rows if x["image"]=="unmatched.jpg")["weather"]=="unknown"
    return {"status":"PASS","images":4,"matched":3,"unmatched_preserved_as_unknown":1}

def main():
    report={"scope":"synthetic readiness only; real training uses scripts/train_maskrcnn_bdd100k.py",
      "instance_bitmask_decoder":test_instance_decoder(),
      "mask_rcnn_forward_backward_and_inference":test_mask_rcnn(),
      "environmental_metadata_join":test_environment_join()}
    report["status"]="PASS" if all(v["status"]=="PASS" for k,v in report.items() if isinstance(v,dict) and "status" in v) else "FAIL"
    Path("component-readiness").mkdir(exist_ok=True)
    Path("component-readiness/report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if report["status"]!="PASS": raise SystemExit(1)
if __name__=="__main__": main()

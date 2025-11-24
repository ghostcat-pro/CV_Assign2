import torch
import numpy as np

@torch.no_grad()
def compute_iou(logits, targets, num_classes=8, ignore_index=None):
    preds = torch.argmax(logits, dim=1)
    ious = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
        p = preds == cls
        t = targets == cls
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        iou = np.nan if union == 0 else inter/union
        ious.append(iou)
    return float(np.nanmean(ious)), ious

@torch.no_grad()
def evaluate_loader(model, loader, device, num_classes=8):
    model.eval()
    miou_all = []
    per_class = [[] for _ in range(num_classes)]
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(imgs)
        if isinstance(out, dict):  # DeepLab
            out = out["out"]
        miou, ious = compute_iou(out, masks, num_classes)
        miou_all.append(miou)
        for c,v in enumerate(ious):
            per_class[c].append(v)
    return float(np.nanmean(miou_all)), [float(np.nanmean(v)) for v in per_class]

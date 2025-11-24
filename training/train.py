import torch
from tqdm import tqdm
from training.eval import compute_iou

def train_one_epoch(model, loader, optimizer, criterion, device, num_classes=8):
    model.train()
    loss_sum, iou_sum = 0.0, 0.0
    for imgs, masks in tqdm(loader, desc="train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        if isinstance(out, dict):  # DeepLab
            out = out["out"]
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        miou, _ = compute_iou(out.detach(), masks, num_classes)
        iou_sum += miou
    return loss_sum/len(loader), iou_sum/len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=8):
    model.eval()
    loss_sum, iou_sum = 0.0, 0.0
    for imgs, masks in tqdm(loader, desc="val", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(imgs)
        if isinstance(out, dict):
            out = out["out"]
        loss = criterion(out, masks)
        loss_sum += loss.item()
        miou, _ = compute_iou(out, masks, num_classes)
        iou_sum += miou
    return loss_sum/len(loader), iou_sum/len(loader)

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
        
        # Handle different model outputs
        if isinstance(out, dict):  # DeepLab
            main_out = out["out"]
            loss = criterion(main_out, masks)
        elif isinstance(out, tuple):  # V4 with deep supervision
            main_out = out[0]  # First element is main output
            loss = criterion(out, masks)  # Pass full tuple to loss
        else:
            main_out = out
            loss = criterion(out, masks)
        
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        miou, _ = compute_iou(main_out.detach(), masks, num_classes)
        iou_sum += miou
    return loss_sum/len(loader), iou_sum/len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=8):
    model.eval()
    loss_sum, iou_sum = 0.0, 0.0
    for imgs, masks in tqdm(loader, desc="val", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(imgs)
        
        # Handle different model outputs
        if isinstance(out, dict):  # DeepLab
            main_out = out["out"]
        elif isinstance(out, tuple):  # Should not happen in eval mode for V4, but handle it
            main_out = out[0]
        else:
            main_out = out
        
        loss = criterion(main_out, masks)
        loss_sum += loss.item()
        miou, _ = compute_iou(main_out, masks, num_classes)
        iou_sum += miou
    return loss_sum/len(loader), iou_sum/len(loader)

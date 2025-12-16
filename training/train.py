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
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_sum += loss.item()
        
        # Clear GPU cache periodically to prevent memory buildup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
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
            loss = criterion(main_out, masks)
        elif isinstance(out, tuple):  # Should not happen in eval mode for V4/V2, but handle it
            main_out = out[0]
            loss = criterion(out, masks)  # Pass full tuple to loss in case criterion handles it
        else:
            main_out = out
            loss = criterion(out, masks)
        
        loss_sum += loss.item()
        miou, _ = compute_iou(main_out, masks, num_classes)
        iou_sum += miou
    return loss_sum/len(loader), iou_sum/len(loader)

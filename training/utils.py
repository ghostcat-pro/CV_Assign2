import os
import torch

def save_checkpoint(model, optimizer, epoch, best_iou, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_iou": best_iou
    }, path)

def load_checkpoint(model, optimizer, path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

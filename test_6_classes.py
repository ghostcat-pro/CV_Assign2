"""
Test that all models can work with 6 classes instead of 8.
Run with: python test_6_classes.py
"""
import torch
from models.unet_resattn import UNetResAttn
from models.unet_resattn_v2 import UNetResAttnV2
from models.unet_resattn_v3 import UNetResAttnV3
from models.suimnet import SUIMNet
from models.deeplab_resnet import get_deeplabv3
from models.uwsegformer import UWSegFormer

def test_model_6_classes(model, name, num_classes=6):
    """Test model with 6 classes using dummy input."""
    model.eval()
    dummy_input = torch.randn(2, 3, 256, 256)  # batch=2, RGB, 256x256
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, dict):  # DeepLab returns dict
                output = output['out']
            
        # Check output shape
        expected_shape = (2, num_classes, 256, 256)  # batch, num_classes, 256x256
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ {name:25s} - Output: {tuple(output.shape)} - Params: {params:,}")
        return True
        
    except Exception as e:
        print(f"✗ {name:25s} - FAILED: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ALL MODELS WITH 6 CLASSES (MERGED MODE)")
    print("=" * 80)
    
    results = []
    num_classes = 6
    
    # Test UNet-ResAttn
    print("\n1. Testing UNet-ResAttn (6 classes)...")
    model1 = UNetResAttn(in_ch=3, out_ch=num_classes, base_ch=64)
    results.append(test_model_6_classes(model1, "UNet-ResAttn", num_classes))
    
    # Test UNet-ResAttn V2
    print("\n2. Testing UNet-ResAttn V2 (6 classes)...")
    model2 = UNetResAttnV2(in_ch=3, out_ch=num_classes, base_ch=64, deep_supervision=True)
    results.append(test_model_6_classes(model2, "UNet-ResAttn V2", num_classes))
    
    # Test UNet-ResAttn V3
    print("\n3. Testing UNet-ResAttn V3 (6 classes)...")
    model3 = UNetResAttnV3(in_ch=3, out_ch=num_classes, pretrained=False)
    results.append(test_model_6_classes(model3, "UNet-ResAttn V3", num_classes))
    
    # Test SUIM-Net
    print("\n4. Testing SUIM-Net (6 classes)...")
    model4 = SUIMNet(in_ch=3, out_ch=num_classes, base=32)
    results.append(test_model_6_classes(model4, "SUIM-Net", num_classes))
    
    # Test DeepLabV3
    print("\n5. Testing DeepLabV3-ResNet50 (6 classes)...")
    model5 = get_deeplabv3(num_classes=num_classes, pretrained=False)
    results.append(test_model_6_classes(model5, "DeepLabV3-ResNet50", num_classes))
    
    # Test UWSegFormer with ResNet-50
    print("\n6. Testing UWSegFormer-ResNet50 (6 classes)...")
    model6 = UWSegFormer(backbone='resnet50', num_classes=num_classes, pretrained=False)
    results.append(test_model_6_classes(model6, "UWSegFormer-ResNet50", num_classes))
    
    # Summary
    print("\n" + "=" * 80)
    if all(results):
        print("✓ ALL TESTS PASSED - All models work correctly with 6 classes!")
    else:
        print(f"✗ SOME TESTS FAILED - {sum(results)}/{len(results)} passed")
    print("=" * 80)
    print("\nModels are ready to train with --merge-classes flag!")

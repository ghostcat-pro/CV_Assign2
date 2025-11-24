"""Quick dry run to test all models can be instantiated and run forward pass."""
import torch
from models.unet_resattn import UNetResAttn
from models.suimnet import SUIMNet
from models.deeplab_resnet import get_deeplabv3

def test_model(model, name):
    """Test model with dummy input."""
    model.eval()
    dummy_input = torch.randn(2, 3, 256, 256)  # batch=2, RGB, 256x256
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, dict):  # DeepLab returns dict
                output = output['out']
            
        # Check output shape
        expected_shape = (2, 8, 256, 256)  # batch, 8 classes, 256x256
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ {name:20s} - Output: {tuple(output.shape)} - Params: {params:,}")
        return True
        
    except Exception as e:
        print(f"✗ {name:20s} - FAILED: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("DRY RUN: Testing Model Instantiation and Forward Pass")
    print("=" * 70)
    
    results = []
    
    # Test UNet-ResAttn (custom model)
    print("\n1. Testing UNet-ResAttn...")
    model1 = UNetResAttn(in_ch=3, out_ch=8, base_ch=64)
    results.append(test_model(model1, "UNet-ResAttn"))
    
    # Test SUIM-Net
    print("\n2. Testing SUIM-Net...")
    model2 = SUIMNet(in_ch=3, out_ch=8, base=32)
    results.append(test_model(model2, "SUIM-Net"))
    
    # Test DeepLabV3 (pretrained=False for quick test)
    print("\n3. Testing DeepLabV3-ResNet50...")
    model3 = get_deeplabv3(num_classes=8, pretrained=False)
    results.append(test_model(model3, "DeepLabV3-ResNet50"))
    
    # Summary
    print("\n" + "=" * 70)
    if all(results):
        print("✓ ALL TESTS PASSED - Models are working correctly!")
    else:
        print(f"✗ SOME TESTS FAILED - {sum(results)}/{len(results)} passed")
    print("=" * 70)

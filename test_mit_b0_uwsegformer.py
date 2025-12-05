"""
Test script to verify MIT-B0 backbone integration with UWSegFormer.
"""
import torch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.uwsegformer import get_uwsegformer


def test_mit_b0_integration():
    """Test MIT-B0 backbone with UWSegFormer."""
    print("="*60)
    print("Testing MIT-B0 Backbone Integration with UWSegFormer")
    print("="*60)

    # Test configuration
    batch_size = 2
    num_classes = 8
    H, W = 256, 192

    # Create model with MIT-B0 backbone
    print("\n1. Creating UWSegFormer with MIT-B0 backbone...")
    model = get_uwsegformer(
        backbone='mit_b0',
        num_classes=num_classes,
        pretrained=False
    )
    print("✓ Model created successfully")

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n2. Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print(f"\n3. Testing forward pass...")
    x = torch.randn(batch_size, 3, H, W)
    print(f"   Input shape: {x.shape}")

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"   Output shape: {output.shape}")

    # Verify output shape
    expected_shape = (batch_size, num_classes, H, W)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print("✓ Output shape is correct")

    # Test with different input sizes
    print(f"\n4. Testing with different input sizes...")
    test_sizes = [(128, 128), (320, 240), (512, 384)]

    for h, w in test_sizes:
        x_test = torch.randn(1, 3, h, w)
        with torch.no_grad():
            out_test = model(x_test)
        expected = (1, num_classes, h, w)
        assert out_test.shape == expected, f"For input {(h,w)}, expected {expected}, got {out_test.shape}"
        print(f"   ✓ Input: {(h,w)} → Output: {out_test.shape}")

    # Test gradient flow
    print(f"\n5. Testing gradient flow...")
    model.train()
    x_grad = torch.randn(1, 3, H, W, requires_grad=True)
    output_grad = model(x_grad)
    loss = output_grad.mean()
    loss.backward()

    # Check if gradients exist
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_gradients, "No gradients computed"
    print("✓ Gradients computed successfully")

    # Compare with ResNet50 baseline
    print(f"\n6. Comparing with ResNet50 baseline...")
    model_resnet = get_uwsegformer(
        backbone='resnet50',
        num_classes=num_classes,
        pretrained=False
    )
    resnet_params = sum(p.numel() for p in model_resnet.parameters())

    print(f"   MIT-B0 parameters:    {total_params:,}")
    print(f"   ResNet50 parameters:  {resnet_params:,}")
    print(f"   Difference:           {total_params - resnet_params:+,} ({(total_params/resnet_params - 1)*100:+.1f}%)")

    # Test inference speed
    print(f"\n7. Testing inference speed...")
    import time

    model.eval()
    x_speed = torch.randn(1, 3, 256, 192)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_speed)

    # Measure
    num_runs = 50
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x_speed)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    fps = 1000 / avg_time
    print(f"   Average inference time: {avg_time:.2f} ms")
    print(f"   FPS: {fps:.2f}")

    print("\n" + "="*60)
    print("✓ All tests passed! MIT-B0 backbone is working correctly.")
    print("="*60)


if __name__ == "__main__":
    test_mit_b0_integration()

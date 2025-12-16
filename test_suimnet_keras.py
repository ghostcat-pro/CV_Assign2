"""
Quick test script to verify SUIM-Net Keras implementation
Tests model instantiation and forward pass
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

try:
    import tensorflow as tf
    import numpy as np
    from models.paper.suimnet import SUIM_Net
    
    print("=" * 70)
    print("Testing SUIM-Net Keras Implementation")
    print("=" * 70)
    
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    
    # Test RSB backbone
    print("\n" + "-" * 70)
    print("Test 1: SUIM-Net with RSB backbone (5 classes)")
    print("-" * 70)
    suimnet_rsb = SUIM_Net(base='RSB', im_res=(320, 240, 3), n_classes=5)
    model_rsb = suimnet_rsb.model
    print(f"✓ Model created successfully")
    print(f"  Input shape: {model_rsb.input_shape}")
    print(f"  Output shape: {model_rsb.output_shape}")
    
    # Test forward pass RSB
    dummy_input_rsb = np.random.rand(1, 240, 320, 3).astype(np.float32)
    output_rsb = model_rsb.predict(dummy_input_rsb, verbose=0)
    print(f"  Forward pass: {dummy_input_rsb.shape} -> {output_rsb.shape}")
    print(f"  Output range: [{output_rsb.min():.3f}, {output_rsb.max():.3f}]")
    
    # Count parameters
    trainable_rsb = sum([tf.keras.backend.count_params(w) for w in model_rsb.trainable_weights])
    print(f"  Parameters: {trainable_rsb:,}")
    
    # Test VGG backbone
    print("\n" + "-" * 70)
    print("Test 2: SUIM-Net with VGG16 backbone (5 classes)")
    print("-" * 70)
    suimnet_vgg = SUIM_Net(base='VGG', im_res=(320, 256, 3), n_classes=5)
    model_vgg = suimnet_vgg.model
    print(f"✓ Model created successfully")
    print(f"  Input shape: {model_vgg.input_shape}")
    print(f"  Output shape: {model_vgg.output_shape}")
    
    # Test forward pass VGG
    dummy_input_vgg = np.random.rand(1, 256, 320, 3).astype(np.float32)
    output_vgg = model_vgg.predict(dummy_input_vgg, verbose=0)
    print(f"  Forward pass: {dummy_input_vgg.shape} -> {output_vgg.shape}")
    print(f"  Output range: [{output_vgg.min():.3f}, {output_vgg.max():.3f}]")
    
    # Count parameters
    trainable_vgg = sum([tf.keras.backend.count_params(w) for w in model_vgg.trainable_weights])
    print(f"  Parameters: {trainable_vgg:,}")
    
    # Test 8 classes
    print("\n" + "-" * 70)
    print("Test 3: SUIM-Net with VGG16 backbone (8 classes)")
    print("-" * 70)
    suimnet_vgg8 = SUIM_Net(base='VGG', im_res=(320, 256, 3), n_classes=8)
    model_vgg8 = suimnet_vgg8.model
    print(f"✓ Model created successfully")
    print(f"  Input shape: {model_vgg8.input_shape}")
    print(f"  Output shape: {model_vgg8.output_shape}")
    
    # Test forward pass with 8 classes
    output_vgg8 = model_vgg8.predict(dummy_input_vgg, verbose=0)
    print(f"  Forward pass: {dummy_input_vgg.shape} -> {output_vgg8.shape}")
    print(f"  Output range: [{output_vgg8.min():.3f}, {output_vgg8.max():.3f}]")
    
    # Test data generator utilities
    print("\n" + "-" * 70)
    print("Test 4: Keras data generator utilities")
    print("-" * 70)
    from data.utils.keras_data_utils import adjustData
    
    # Create dummy image and mask
    dummy_img = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    dummy_mask = np.zeros((240, 320, 3), dtype=np.uint8)
    # Add some colored pixels (SUIM colors)
    dummy_mask[50:100, 50:100] = [0, 0, 255]  # Blue - Human diver
    dummy_mask[100:150, 100:150] = [0, 255, 0]  # Green - Plants
    
    # Test with 5 classes
    adj_img_5, adj_mask_5 = adjustData(dummy_img, dummy_mask, n_classes=5)
    print(f"✓ adjustData (5 classes) working")
    print(f"  Input: img {dummy_img.shape}, mask {dummy_mask.shape}")
    print(f"  Output: img {adj_img_5.shape}, mask {adj_mask_5.shape}")
    print(f"  Image range: [{adj_img_5.min():.3f}, {adj_img_5.max():.3f}]")
    print(f"  Mask sum per class: {adj_mask_5.sum(axis=(0,1))}")
    
    # Test with 8 classes
    adj_img_8, adj_mask_8 = adjustData(dummy_img, dummy_mask, n_classes=8)
    print(f"✓ adjustData (8 classes) working")
    print(f"  Output: img {adj_img_8.shape}, mask {adj_mask_8.shape}")
    print(f"  Mask sum per class: {adj_mask_8.sum(axis=(0,1))}")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nYou can now train SUIM-Net with:")
    print("  python train_suimnet_keras.py --base VGG --epochs 50 --batch_size 8")
    print("  python train_suimnet_keras.py --base RSB --epochs 50 --batch_size 8")
    
except ImportError as e:
    print("\n" + "=" * 70)
    print("ERROR: Missing dependencies")
    print("=" * 70)
    print(f"\n{e}")
    print("\nPlease install TensorFlow and Keras:")
    print("  pip install tensorflow==2.10.0 keras==2.10.0")
    print("\nOr install all requirements:")
    print("  pip install -r requirements.txt")
    
except Exception as e:
    print("\n" + "=" * 70)
    print("ERROR during testing")
    print("=" * 70)
    print(f"\n{e}")
    import traceback
    traceback.print_exc()

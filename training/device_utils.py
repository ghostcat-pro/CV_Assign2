"""
Device utilities for cross-platform GPU/CPU detection.

Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback.
"""
import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Get the best available device for PyTorch.

    Priority:
    1. CUDA (NVIDIA GPU) - Windows/Linux
    2. MPS (Apple Silicon GPU) - macOS M1/M2/M3
    3. CPU - Fallback

    Args:
        prefer_mps: If True and MPS is available, use it. Set to False
                   to use CPU even if MPS is available (for debugging).

    Returns:
        torch.device: The selected device

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
        >>> data = data.to(device)
    """
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if prefer_mps:
            device = torch.device('mps')
            print("Using device: MPS (Apple Silicon)")
            return device
        else:
            print("MPS available but not using (prefer_mps=False)")

    # Fallback to CPU
    device = torch.device('cpu')
    print("Using device: CPU")
    print("  Warning: Training on CPU will be slow. Consider using a GPU.")
    return device


def get_device_info():
    """
    Print detailed information about available devices.
    """
    print("=" * 60)
    print("Device Information")
    print("=" * 60)

    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # CUDA info
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")

    # MPS info (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print(f"\nMPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("  Device: Apple Silicon GPU")

    # CPU info
    print(f"\nCPU threads: {torch.get_num_threads()}")

    print("=" * 60)


def move_to_device(data, device):
    """
    Recursively move data to device.

    Handles tensors, lists, tuples, and dictionaries.

    Args:
        data: Data to move (tensor, list, tuple, dict)
        device: Target device

    Returns:
        Data on the target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        return data


if __name__ == "__main__":
    # Test device detection
    get_device_info()
    print("\nSelected device:")
    device = get_device()

    # Test tensor creation
    print("\nTesting tensor creation:")
    x = torch.randn(2, 3, 224, 224, device=device)
    print(f"  Tensor shape: {x.shape}")
    print(f"  Tensor device: {x.device}")
    print("\n✓ Device utilities working correctly!")

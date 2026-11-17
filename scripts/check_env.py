"""Check that PyTorch sees your GPU and that mixed precision works.

    python scripts/check_env.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from srl.utils import env_info  # noqa: E402


def main():
    info = env_info()
    for k in ("python", "torch", "torchvision", "numpy", "cuda_available", "gpu", "cudnn", "git"):
        print(f"{k:15s} {info.get(k)}")
    if not torch.cuda.is_available():
        print("\nPyTorch cannot see a GPU. Reinstall PyTorch with CUDA support "
              "(see README, 'Local setup'), then rerun this script.")
        sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    print(f"{'gpu memory':15s} {props.total_memory / 2**30:.1f} GB")
    from torchvision.models import resnet50
    model = resnet50(weights=None).cuda().to(memory_format=torch.channels_last)
    x = torch.randn(32, 3, 224, 224, device="cuda").contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(x).float().mean()
    loss.backward()
    torch.cuda.synchronize()
    print(f"{'amp smoke test':15s} ok (ResNet-50 forward/backward, batch 32, "
          f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB)")


if __name__ == "__main__":
    main()

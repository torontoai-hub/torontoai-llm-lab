# GPU environment check
import torch
import platform
import subprocess
import os

print("="*70)
print("SYSTEM INFORMATION")
print("="*70)
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    print("\n" + "="*70)
    print("GPU DETAILS")
    print("="*70)

    total_memory_gb = 0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        total_memory_gb += memory_gb

        print(f"\nGPU[{i}]:")
        print(f"  Name: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Memory: {memory_gb:.2f} GB")
        print(f"  Multi-Processors: {props.multi_processor_count}")

        # Check if suitable for DeepSeek-V3.2
        if "H200" in props.name:
            print(f"  Status: ✅ Hopper architecture - Supported")
        elif "B200" in props.name or "GB200" in props.name:
            print(f"  Status: ✅ Blackwell architecture - Optimal")
        else:
            print(f"  Status: ⚠️ May not be supported")

    print(f"\nTotal GPU Memory: {total_memory_gb:.2f} GB")

    # Check NVLink connectivity
    print("\n" + "="*70)
    print("NVLINK STATUS")
    print("="*70)
    try:
        nvlink_output = subprocess.check_output(['nvidia-smi', 'nvlink', '--status'],
                                                stderr=subprocess.STDOUT, text=True)
        print("✅ NVLink detected - Multi-GPU performance will be optimal")
    except:
        print("⚠️ Could not detect NVLink - Performance may be degraded")

    # Recommendations
    print("\n" + "="*70)
    print("CONFIGURATION RECOMMENDATIONS")
    print("="*70)

    if total_memory_gb >= 1100:
        print("✅ Sufficient GPU memory for DeepSeek-V3.2-Exp")
        print("   Recommended mode: EP/DP (--dp 8 --enable-expert-parallel)")
    elif total_memory_gb >= 900:
        print("⚠️ Marginal GPU memory - Consider using FP8 quantization")
        print("   Recommended mode: Tensor Parallel (--tensor-parallel-size 8)")
    else:
        print("❌ Insufficient GPU memory for full model")
        print("   Consider: Smaller model or quantized version")

else:
    print("❌ CUDA not available - GPU required for this model")

````markdown
# DeepSeek V3.2 Experimental Setup

This document provides setup instructions for running the DeepSeek V3.2 Experimental model using vLLM with multi-GPU support on a CUDA-enabled system as of 8 Oct 2025

---

## Environment Setup

### 1. Set Distro and Architecture
```bash
export distro=ubuntu2404
export arch=x86_64
````

### 2. Add NVIDIA CUDA Repository Keyring

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

### 3. (Optional) Add CUDA Archive Key Manually

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-archive-keyring.gpg
sudo mv cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/ /" | sudo tee /etc/apt/sources.list.d/cuda-$distro-$arch.list
```

### 4. Install NVIDIA Drivers

```bash
sudo apt update
sudo apt install -y cuda-drivers
```

### 5. Verify Installation

```bash
nvidia-smi
```

---

## Python and Environment Setup

### 6. Install Python and Pip

```bash
sudo apt install -y python3 python3-pip
pip install --upgrade pip
```

### 7. Install UV Package Manager (Optional)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 8. Create and Activate Virtual Environment

```bash
uv venv
source .venv/bin/activate
```

---

## Install Dependencies

### 9. Install PyTorch Nightly with CUDA 13.0 Support

```bash
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

Check installation:

```python
import torch
torch.cuda.is_available()
```

### 10. Install vLLM and DeepGemm

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
uv pip install https://wheels.vllm.ai/dsv32/deep_gemm-2.1.0%2B594953a-cp312-cp312-linux_x86_64.whl
```

### 11. Install Supporting Python Libraries

```bash
uv pip install openai transformers accelerate numpy --quiet
```

### 12. Verify Environment

```bash
python -c "import torch, vllm, transformers, numpy; print('Environment ready')"
```

---

## Run DeepSeek Model with vLLM

```
# Tensor Parallel - Better for H200
vllm serve deepseek-ai/DeepSeek-V3.2-Exp -tp 8
```

```bash
vllm serve deepseek-ai/DeepSeek-V3.2-Exp \
  -dp 8 \
  --enable-expert-parallel \
  --served-model-name deepseek-v32 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code
```

---

## System Validation Example

```bash
python3 system_validation.py
```

Example output:

```
======================================================================
SYSTEM INFORMATION
======================================================================
OS: Linux 6.8.0-79-generic
Python: 3.12.3
PyTorch: 2.8.0+cu128
CUDA available: True
CUDA version: 12.8
cuDNN version: 91002
Number of GPUs: 8

======================================================================
GPU DETAILS
======================================================================
GPU[0-7]: NVIDIA H200 (150.11 GB each, Compute Capability: 9.0)
Hopper architecture - Supported

Total GPU Memory: 1200.88 GB

======================================================================
NVLINK STATUS
======================================================================
NVLink detected - Multi-GPU performance will be optimal

======================================================================
CONFIGURATION RECOMMENDATIONS
======================================================================
Sufficient GPU memory for DeepSeek-V3.2-Exp
Recommended mode: EP/DP (--dp 8 --enable-expert-parallel)
```

---

## DeepGemm FP8 Warmup Logs

```
DeepGemm(fp8_gemm_nt) warmup (W=torch.Size([4096, 7168])): 100%|██████████| 8192/8192 [01:01<00:00, 133.63it/s]
DeepGemm(m_grouped_fp8_gemm_nt_contiguous) warmup (W=torch.Size([32, 4096, 7168])): 100%|███████| 2080/2080 [01:22<00:00, 25.11it/s]
...
```

---

## Evaluation with LM-Eval

### Run GSM8K Benchmark (5-shot)

```bash
lm-eval --model local-completions --tasks gsm8k \
  --model_args model=deepseek-ai/DeepSeek-V3.2-Exp,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=100,max_retries=3,tokenized_requests=False
```

Output:

| Tasks | Version | Filter           | n-shot | Metric      | ↑ | Value  | ± | Stderr |
| ----- | ------- | ---------------- | ------ | ----------- | - | ------ | - | ------ |
| gsm8k | 3       | flexible-extract | 5      | exact_match | ↑ | 0.9507 | ± | 0.0060 |
|       |         | strict-match     | 5      | exact_match | ↑ | 0.9484 | ± | 0.0061 |

---

### Run GSM8K Benchmark (20-shot)

```bash
lm-eval --model local-completions --tasks gsm8k \
  --model_args model=deepseek-ai/DeepSeek-V3.2-Exp,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=100,max_retries=3,tokenized_requests=False \
  --num_fewshot 20
```

Output:

| Tasks | Version | Filter           | n-shot | Metric      | ↑ | Value  | ± | Stderr |
| ----- | ------- | ---------------- | ------ | ----------- | - | ------ | - | ------ |
| gsm8k | 3       | flexible-extract | 20     | exact_match | ↑ | 0.9416 | ± | 0.0065 |
|       |         | strict-match     | 20     | exact_match | ↑ | 0.9393 | ± | 0.0066 |

---

### Deploy the Prometheus Grafana Stack

```
docker compose up -d
```

## Summary

* Model: deepseek-ai/DeepSeek-V3.2-Exp
* Framework: vLLM Nightly Build with DeepGemm FP8 Optimizations
* Environment: Ubuntu 24.04 + CUDA 13.0 + NVIDIA H200 GPUs
* GSM8K (5-shot): 95.07%
* GSM8K (20-shot): 94.16%
* All GPUs connected via NVLink for optimal expert-parallel performance.

---

## References

* [vLLM GitHub](https://github.com/vllm-project/vllm)
* [DeepSeek AI Models](https://huggingface.co/deepseek-ai)
* [LM-Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness)



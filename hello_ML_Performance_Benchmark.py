#!/usr/bin/env python
# coding: utf-8

# # ML Training Throughput Benchmark (run in Colab or Locally)
# 
# Measures synthetic training throughput (images/sec) for small CNNs in PyTorch and TensorFlow. Avoids disk I/O to focuse on compute performance.
# 
# - Mixed precision AMP for NVIDIA GPUs
# - Increase `BATCH_SIZE` or switch to `resnet50` for a heavier test
# - Dependencies: pip install -q torch torchvision tensorflow tqdm
# 

#%%---------------------------------------------------------------------------
# Set Configuration

USE_GPU      = True       # Set False to force CPU even if CUDA is available
PRECISION    = "fp32"      # "fp32" | "amp" (CUDA mixed precision) | "bf16" (if supported)
BATCH_SIZE   = 64
IMAGE_SIZE   = 224
ITERS        = 200        # Timed iterations
WARMUP       = 10         # Warmup iterations (not timed)
MODEL_NAME   = "resnet50" # "resnet18" (fast) or "resnet50" (heavier)
print(f"Config => USE_GPU={USE_GPU}, PRECISION={PRECISION}, BATCH_SIZE={BATCH_SIZE},\n"
      f"IMAGE_SIZE={IMAGE_SIZE}, ITERS={ITERS}, WARMUP={WARMUP}, MODEL={MODEL_NAME}")


#%%---------------------------------------------------------------------------
# PyTorch benchmark

import time, os, math
import torch
import torch.nn as nn
import torch.optim as optim


if USE_GPU:
    if torch.cuda.is_available():
        print("Using CUDA")
        # Get the name of the current default GPU
        GPU_NAME = torch.cuda.get_device_name(0) # Index 0 for the first GPU
        print(f"GPU Name: {GPU_NAME}")
    
        # Get more detailed properties
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"Total Memory: {GPU_MEMORY:.0f} MB")

# Try to provide a progress bar; fall back to prints if tqdm not installed
try:
    from tqdm.auto import tqdm
    def progress_iter(it, total):
        return tqdm(range(it), total=total)
except Exception:
    def progress_iter(it, total):
        class Dummy:
            def __init__(self, it): self.it = it
            def __iter__(self):
                for i in range(self.it):
                    if i % max(1, self.it // 10) == 0:
                        print(f"Progress: {i}/{self.it}")
                    yield i
        return Dummy(it)

def get_device(use_gpu: bool) -> str:
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def make_model(name="resnet18", num_classes=1000):
    try:
        from torchvision.models import resnet18, resnet50
        if name == "resnet18":
            print(f"Using model: {name}")
            return resnet18(num_classes=num_classes)
        elif name == "resnet50":
            print(f"Using model: {name}")
            return resnet50(num_classes=num_classes)
        else:
            raise ValueError("Supported: resnet18, resnet50")
    except Exception:
        # Fallback tiny CNN if torchvision isn’t available
        print("Using model: Small Dummy")
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1000),
        )

device = get_device(USE_GPU)
print("Device:", device)
if device == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | VRAM: {p.total_memory/1024**3:.1f} GB | CC: {p.major}.{p.minor}")

torch.backends.cudnn.benchmark = True  # speedup for fixed shapes

# Model / loss / opt
model = make_model(MODEL_NAME).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Channels-last helps CNNs on CUDA, esp. with AMP/BF16
if device == "cuda" and PRECISION in ("amp", "bf16"):
    model = model.to(memory_format=torch.channels_last)

# Synthetic batch
N, C, H, W = BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE
x = torch.randn(N, C, H, W, device=device)
y = torch.randint(0, 1000, (N,), device=device)
if device == "cuda" and PRECISION in ("amp", "bf16"):
    x = x.to(memory_format=torch.channels_last)

# Precision helpers
use_amp  = (device == "cuda" and PRECISION == "amp")
use_bf16 = (device == "cuda" and PRECISION == "bf16")
scaler   = torch.cuda.amp.GradScaler(enabled=use_amp)
bf16_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else None

def train_step():
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        with torch.cuda.amp.autocast():
            out = model(x); loss = criterion(out, y)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    elif use_bf16:
        with bf16_ctx:
            out = model(x); loss = criterion(out, y)
        loss.backward(); optimizer.step()
    else:
        out = model(x); loss = criterion(out, y)
        loss.backward(); optimizer.step()

# Warmup (with a tiny progress signal)
print(f"Warmup: {WARMUP} iters...")
for i in progress_iter(WARMUP, WARMUP):
    train_step()
if device == "cuda":
    torch.cuda.synchronize()

# Timed section (with progress bar)
print(f"Timed run: {ITERS} iters...")
t0 = time.perf_counter()
for _ in progress_iter(ITERS, ITERS):
    train_step()
if device == "cuda":
    torch.cuda.synchronize()
dt = time.perf_counter() - t0

imgs = ITERS * BATCH_SIZE
ips  = imgs / dt
print("\n=== PYTORCH RESULTS ===")
print(f"Device: {device} | Precision: {PRECISION}")
if USE_GPU:
        print(f"GPU Name: {GPU_NAME} w/ memory {GPU_MEMORY:.0f} MB")
print(f"Model: {MODEL_NAME} | Batch: {BATCH_SIZE} | Image: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Iters: {ITERS} (warmup {WARMUP}) | Time: {dt:.3f}s")
print(f"Throughput: {ips:,.1f} images/sec  (synthetic data)")



#%%---------------------------------------------------------------------------
# TensorFlow benchmark (Need to run in another environment with TF installed)

import time, os
try:
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    tf_ok = True
except Exception as e:
    tf_ok = False
    print("TensorFlow not available; install tensorflow to run this cell.")

if tf_ok:
    # Device selection: if USE_GPU False or no GPU, force CPU
    dev_name = "/GPU:0" if (USE_GPU and len(tf.config.list_physical_devices('GPU'))>0) else "/CPU:0"
    print("TF device:", dev_name)

    # Mixed precision policy (only meaningful on GPU)
    if "GPU" in dev_name and PRECISION in ("amp", "bf16"):
        policy = "mixed_bfloat16" if PRECISION=="bf16" else "mixed_float16"
        try:
            mixed_precision.set_global_policy(policy)
            print("TF mixed precision:", policy)
        except Exception as e:
            print("Could not set mixed precision:", e)

    # Simple Keras model (ResNet50) if available; fallback to small CNN
    try:
        model = tf.keras.applications.ResNet50(weights=None, classes=1000)
        model_name = "ResNet50"
    except Exception:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu', input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1000)
        ])
        model_name = "TinyCNN"

    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    # Synthetic inputs
    x = tf.random.normal([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.random.uniform([BATCH_SIZE], maxval=1000, dtype=tf.int32)

    # Warmup with progress
    print(f"Warmup: {WARMUP} iters...")
    for i in range(WARMUP):
        model.train_on_batch(x, y)
        if (i+1) % max(1, WARMUP//10) == 0:
            print(f"Warmup progress: {i+1}/{WARMUP}")

    # Timed with progress
    print(f"Timed run: {ITERS} iters...")
    t0 = time.perf_counter()
    for i in range(ITERS):
        model.train_on_batch(x, y)
        if (i+1) % max(1, ITERS//10) == 0:
            print(f"Run progress: {i+1}/{ITERS}")
    dt = time.perf_counter() - t0

    ips = (ITERS * BATCH_SIZE) / dt
    print("\n=== TENSORFLOW RESULTS ===")
    print(f"Device: {dev_name} | Precision: {PRECISION}")
    print(f"Model: {model_name} | Batch: {BATCH_SIZE} | Image: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Iters: {ITERS} (warmup {WARMUP}) | Time: {dt:.3f}s")
    print(f"Throughput: {ips:,.1f} images/sec  (synthetic data)")


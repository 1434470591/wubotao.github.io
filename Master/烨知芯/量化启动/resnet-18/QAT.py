
import os
import math
import random
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision import transforms

# 兼容不同 torchvision 版本
try:
    from torchvision.models import ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
except Exception:
    weights = None

# ImageNet 兜底（几乎所有 ResNet 预处理都用这组）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# 优先从 weights.transforms() 里拿到 Normalize 的 mean/std
try:
    _eval_tf = weights.transforms() if weights is not None else None
    norm = _eval_tf.transforms[-1]
    mean = getattr(norm, 'mean', IMAGENET_MEAN)
    std  = getattr(norm, 'std',  IMAGENET_STD)
except Exception:
    _eval_tf = None
    mean, std = IMAGENET_MEAN, IMAGENET_STD

# ------------------------------
# Config
# ------------------------------
SEED = 0
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
DATA_DIR = os.getenv("DATA_DIR", "./data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # full QAT on GPU if available
EPOCHS = int(os.getenv("EPOCHS", "5"))  # QAT epochs (tune as needed)
LR = float(os.getenv("LR", "1e-3"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "5e-4"))
MOMENTUM = float(os.getenv("MOMENTUM", "0.9"))
FREEZE_BACKBONE = os.getenv("FREEZE_BACKBONE", "0") == "1"  # QAT通常建议全模型微调，这里提供选项

# ------------------------------
# Reproducibility
# ------------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False  # allow faster kernels
    cudnn.benchmark = True

set_seed(SEED)

# ------------------------------
# Model
# ------------------------------
try:
    from torchvision.models import resnet18, ResNet18_Weights
    _RESNET_W = ResNet18_Weights.DEFAULT
except Exception:
    # Older torchvision fallback
    from torchvision.models import resnet18
    _RESNET_W = None

def build_model(num_classes=10):
    m = resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ------------------------------
# Data
# ------------------------------
# Keep ImageNet-like preprocessing to stay consistent with pretrained weights.
# CIFAR-10 will be resized to 224x224.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

test_transform = _eval_tf or transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
test_set  = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

# ------------------------------
# PT2E Quantization (QAT via FakeQuant)
# ------------------------------
# We implement a custom Quantizer similar to the user's PTQ quantizer, but using FakeQuantize.
from torch.ao.quantization.quantizer import (
    QuantizationSpec,
    QuantizationAnnotation,
)
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e

import torch
from torch.fx import GraphModule
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver, MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver
)
from torch.ao.quantization.fake_quantize import FakeQuantize

aten = torch.ops.aten

def _is_conv2d(n):  return n.op == "call_function" and n.target == aten.conv2d.default
def _is_linear(n):  return n.op == "call_function" and n.target == aten.linear.default
def _is_relu(n):    return n.op == "call_function" and n.target in (aten.relu_.default, aten.relu.default)
def _is_maxpool(n): return n.op == "call_function" and n.target in (aten.max_pool2d.default, aten.max_pool2d_with_indices.default)
def _is_adapool(n): return n.op == "call_function" and n.target == aten.adaptive_avg_pool2d.default
def _is_flatten(n): return n.op == "call_function" and n.target == aten.flatten.using_ints
def _is_add(n):     return n.op == "call_function" and n.target in (aten.add.Tensor, aten.add_.Tensor)

# =============== QAT 版 Quantizer ===============
class MyQATQuantizer(Quantizer):
    def __init__(self, per_channel_weight: bool = True, use_hist_for_act: bool = False):
        super().__init__()
        # --- 激活：uint8/affine/FakeQuant ---
        act_obs = (HistogramObserver.with_args(bins=1024)
                   if use_hist_for_act else
                   MovingAverageMinMaxObserver.with_args(averaging_constant=0.01))
        self.act_qspec = QuantizationSpec(
            dtype=torch.uint8,
            quant_min=0, quant_max=255,
            qscheme=torch.per_tensor_affine,
            is_dynamic=False,
            # 关键：换成 FakeQuantize
            observer_or_fake_quant_ctr=FakeQuantize.with_args(observer=act_obs),
        )

        # --- 权重：int8/对称/FakeQuant ---
        if per_channel_weight:
            self.w_qspec = QuantizationSpec(
                dtype=torch.int8,
                quant_min=-128, quant_max=127,
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,  # Conv/Linear 的 out_channel 轴
                is_dynamic=False,
                observer_or_fake_quant_ctr=FakeQuantize.with_args(
                    observer=PerChannelMinMaxObserver.with_args(reduce_range=False)
                ),
            )
        else:
            self.w_qspec = QuantizationSpec(
                dtype=torch.int8,
                quant_min=-128, quant_max=127,
                qscheme=torch.per_tensor_symmetric,
                is_dynamic=False,
                observer_or_fake_quant_ctr=FakeQuantize.with_args(
                    observer=MinMaxObserver.with_args(reduce_range=False)
                ),
            )

    def annotate(self, gm: GraphModule) -> GraphModule:
        for n in gm.graph.nodes:
            if _is_conv2d(n) or _is_linear(n):
                x = n.args[0]
                qa = QuantizationAnnotation(
                    input_qspec_map={x: self.act_qspec},
                    output_qspec=self.act_qspec,
                    _annotated=True,
                )
                n.meta["quantization_annotation"] = qa
                # 权重量化（注意：权重是第2个 arg）
                if isinstance(n.args[1], torch.fx.Node):
                    wqa = QuantizationAnnotation(
                        input_qspec_map={},
                        output_qspec=self.w_qspec,
                        _annotated=True,
                    )
                    n.args[1].meta["quantization_annotation"] = wqa

            elif _is_relu(n) or _is_maxpool(n) or _is_adapool(n) or _is_flatten(n):
                qa = QuantizationAnnotation(
                    input_qspec_map={},
                    output_qspec=self.act_qspec,
                    _annotated=True,
                )
                n.meta["quantization_annotation"] = qa

            elif _is_add(n):
                a, b = n.args[0], n.args[1]
                qa = QuantizationAnnotation(
                    input_qspec_map={a: self.act_qspec, b: self.act_qspec},
                    output_qspec=self.act_qspec,
                    _annotated=True,
                )
                n.meta["quantization_annotation"] = qa

        return gm

    # ---- validate：简单覆盖率检查（帮助发现漏标）----
    def validate(self, gm: GraphModule) -> None:
        must_cover = []
        for n in gm.graph.nodes:
            if (_is_conv2d(n) or _is_linear(n) or _is_relu(n) or _is_maxpool(n)
                or _is_add(n) or _is_adapool(n) or _is_flatten(n)):
                must_cover.append(n)
        missed = [n for n in must_cover if "quantization_annotation" not in n.meta]
        if missed:
            raise RuntimeError(f"Missed nodes: {[str(m.target) for m in missed]}")


# ------------------------------
# Export to ATen graph
# ------------------------------
def export_to_graphmodule(model: nn.Module, device: str) -> torch.fx.GraphModule:
    model.eval()
    example = (torch.randn(1, 3, 224, 224, device=device),)
    # torch.export returns ExportedProgram; .module() gives a GraphModule
    exported = torch.export.export(model, args=example, strict=False).module()
    return exported

# ------------------------------
# Train/Eval loops
# ------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: str, epoch: int):
    torch.ao.quantization.move_exported_model_to_train(model)
    total, correct, running_loss = 0, 0, 0.0
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    total, correct = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return correct / total * 100.0

# =============== 仅微调最后一层 ===============
def quick_finetune(model, train_loader, epochs=2, device=DEVICE):
    if epochs <= 0:
        return model
    # 冻结 backbone，只训练最后一层，全程很快
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.fc.parameters():
        p.requires_grad_(True)

    model.to(device).train()
    opt = torch.optim.AdamW(model.fc.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    for ep in range(epochs):
        t0, running, n = time.time(), 0.0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = ce(logits, labels)
            loss.backward()
            opt.step()
            running += loss.item()
            n += 1
        print(f"[finetune] epoch {ep+1}/{epochs}  loss={running/max(1,n):.4f}  time={time.time()-t0:.1f}s")
    return model.cpu().eval()

# ------------------------------
# Main
# ------------------------------
def main():
    print(f"[Config] DEVICE={DEVICE}, EPOCHS={EPOCHS}, LR={LR}, FREEZE_BACKBONE={FREEZE_BACKBONE}")
    # 1) Build and (optionally) freeze backbone
    model = build_model(num_classes=10)

    # 2) Move to GPU for export example and keep parameters on GPU
    model = model.to(DEVICE)

    # fine-tune最后一层，确保模型精度
    model = quick_finetune(model, train_loader, epochs=2, device=DEVICE)
    model.eval()
    base_acc = evaluate(model.to(DEVICE), test_loader, device=DEVICE)
    print(f"[fp32] Top-1 = {base_acc:.2f}%")

    # 3) Export to ATen graph
    print("[Step] Export model to ATen graph...")
    gm_fp = export_to_graphmodule(model, DEVICE)

    # 4) Prepare QAT with FakeQuant
    print("[Step] Prepare QAT (FakeQuant)...")
    qat_quantizer = MyQATQuantizer()
    prepared = prepare_pt2e(gm_fp, qat_quantizer)

    # Ensure prepared model on GPU
    prepared = prepared.to(DEVICE)

    # 5) QAT training on GPU
    print("[Step] Start QAT training...")
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    params = [p for p in prepared.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(prepared, train_loader, criterion, optimizer, DEVICE, epoch)
        torch.ao.quantization.move_exported_model_to_eval(prepared)
        test_acc = evaluate(prepared, test_loader, DEVICE)
        best_acc = max(best_acc, test_acc)
        print(f"[Epoch {epoch+1:02d}/{EPOCHS}] train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% test_acc(FQ)={test_acc:.2f}% best={best_acc:.2f}%")

    # 6) Convert to int8
    print("[Step] Converting to int8 (CPU backend)...")
    quantized = convert_pt2e(prepared)
    # Move to CPU for int8 inference
    quantized = quantized.to(DEVICE)
    torch.ao.quantization.move_exported_model_to_eval(quantized)
    cpu_acc = evaluate(quantized, test_loader, device=DEVICE)
    print(f"[INT8-CPU] Top-1 Acc: {cpu_acc:.2f}%")

if __name__ == "__main__":
    main()

import os, time, math
import torch
import torch.nn as nn
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from torch.export import export
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# =============== 超参数 ===============
DATA_DIR = "./data"
BATCH_SIZE = 128
NUM_WORKERS = min(8, os.cpu_count() or 1)
EPOCHS = 2                 # 设为 0 可跳过微调，直接做 PTQ
LR = 1e-3
CALIB_BATCHES = None       # 设为 N 可只用前 N 个 batch 做校准；None = 全量训练集
DEVICE_TRAIN = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_EVAL = "cpu"        # XNNPACK 目标为 CPU，导出/量化和评测均在 CPU

torch.manual_seed(0)

# =============== 兼容导出与量化 API（新优先，旧回退） ===============
USE_TORCHAO = False
USE_EXECUTORCH_XNNPACK = False

# =============== 数据集（CIFAR-10） ===============
# 使用 ImageNet 归一化与 224 尺寸以贴合 ResNet-18 预训练权重
from torchvision.models import resnet18
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
if weights is not None:
    _eval_tf = weights.transforms()                # 官方推荐的 eval 预处理
    _norm = next((t for t in getattr(_eval_tf, "transforms", [])
                  if isinstance(t, transforms.Normalize)), None)
    if _norm is not None:
        mean, std = tuple(_norm.mean), tuple(_norm.std)
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
else:
    _eval_tf = None
    mean, std = IMAGENET_MEAN, IMAGENET_STD


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
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# =============== 模型：ResNet-18（替换最后一层为 10 类） ===============
def build_model():
    m = resnet18(weights=weights)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, 10)
    return m

# =============== 评测（Top-1） ===============
@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred.cpu() == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100.0

# =============== 仅微调最后一层（可选） ===============
def quick_finetune(model, epochs=2, device=DEVICE_TRAIN):
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

# =============== 导出为 PT2E 图（兼容新旧 API） ===============
def export_for_pt2e(model, example_inputs):
    model = model.eval().cpu()
    return export(model, example_inputs).module()

# =============== 量化（XNNPACK PTQ） ===============
def quantize_pt2e_xnnpack(exported_model, calib_loader):
    # 量化器配置：对称量化，按需可以切换 is_per_channel=True
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=False)
    )

    prepared = prepare_pt2e(exported_model, quantizer)

    # 校准：用全量训练集前向一遍（只需 CPU）
    torch.ao.quantization.move_exported_model_to_eval(prepared)
    with torch.inference_mode():
        seen = 0
        for i, (images, _) in enumerate(calib_loader):
            prepared(images.cpu())
            seen += 1
            if CALIB_BATCHES is not None and seen >= CALIB_BATCHES:
                break

    quantized = convert_pt2e(prepared)
    return quantized

# =============== 主流程 ===============
def main():
    # 1) 构建并（可选）微调
    float_model = build_model()
    float_model = quick_finetune(float_model, epochs=EPOCHS, device=DEVICE_TRAIN)
    base_acc = evaluate(float_model.to(DEVICE_EVAL), test_loader, device=DEVICE_EVAL)
    print(f"[fp32 baseline] Top-1 = {base_acc:.2f}%")

    # 2) 导出 PT2E
    example_inputs = (torch.randn(1, 3, 224, 224),)
    exported_fp32 = export_for_pt2e(float_model, example_inputs)

    # 3) 量化（XNNPACK）
    quantized_model = quantize_pt2e_xnnpack(exported_fp32, train_loader)

    # 4) 评测量化模型（CPU）
    q_acc = evaluate(quantized_model, test_loader, device=DEVICE_EVAL)
    print(f"[int8 (PT2E+XNNPACK)] Top-1 = {q_acc:.2f}%")

    # 5) 可选：保存
    torch.save(float_model.state_dict(), "resnet18_cifar10_fp32.pth")
    torch.save(quantized_model.state_dict(), "resnet18_cifar10_pt2e_xnnpack_int8.pth")
    print("Saved: fp32 -> resnet18_cifar10_fp32.pth ; int8 -> resnet18_cifar10_pt2e_xnnpack_int8.pth")

if __name__ == "__main__":
    main()

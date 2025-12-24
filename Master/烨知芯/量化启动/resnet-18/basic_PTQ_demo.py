import os
import time
import warnings
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torchao
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torchvision.models import resnet18

# ---------- 环境配置 ----------
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r".*")
warnings.filterwarnings("default", module=r"torchao.quantization.pt2e")
torch.manual_seed(191009)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 工具函数 ----------
class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name, self.fmt = name, fmt
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            res.append(correct[:k].reshape(-1).float().sum(0).mul_(100.0 / target.size(0)))
        return res

# ---------- 数据 ----------
def prepare_data_loaders(root, train_bs=32, eval_bs=32):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm,
    ])
    train_set = ImageNet(root, split="train", transform=train_tf)
    val_set = ImageNet(root, split="val", transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=train_bs,
                              shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=eval_bs,
                            shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader

def evaluate(model, criterion, loader, max_batches=None):
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    torchao.quantization.pt2e.move_exported_model_to_eval(model)
    with torch.no_grad():
        for i, (img, target) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            img, target = img.to(device), target.to(device)
            output = model(img)
            acc1, acc5 = accuracy(output, target, (1, 5))
            top1.update(acc1.item(), img.size(0))
            top5.update(acc5.item(), img.size(0))

    return top1, top5

# ---------- 主流程 ----------
def main():
    data_root = str(Path("./data/imagenet").expanduser())
    pt_dir = Path("data")
    pt_dir.mkdir(exist_ok=True)
    float_ckpt = "resnet18-pretrained_float.pth"

    eval_batches = None
    batch_size = 32

    # train_loader, val_loader = prepare_data_loaders(data_root, train_bs=batch_size, eval_bs=batch_size)

    # 加载 float 模型
    model_fp32 = resnet18(weights=None)
    if float_ckpt.exists():
        state_dict = torch.load(float_ckpt, map_location="cpu")
        model_fp32.load_state_dict(state_dict)
    else:
        print("float_ckpt 文件不存在，请检查路径或使用 torchvision 预训练权重")
        return
    model_fp32.eval().to(device)

    # 导出 PT2E 模型
    example_inputs = (torch.randn(2, 3, 224, 224, device=device),)
    exported_program = torch.export.export(model_fp32, example_inputs)
    exported_model = exported_program.module()

    print("exported:", exported_model)

    # 设置静态量化配置
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        get_symmetric_quantization_config, XNNPACKQuantizer
    )
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_qat=False))

    # 准备模型（静态量化）
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    prepared_model = prepare_pt2e(exported_model, quantizer)

    print("prepared:", prepared_model)
    prepared_model.to(device)
    torchao.quantization.pt2e.move_exported_model_to_eval(prepared_model)

    # # ---------- 校准 ----------
    # print("📊 Calibrating with full training set...")
    # with torch.no_grad():
    #     for i, (img, _) in enumerate(train_loader):
    #         img = img.to(device)
    #         prepared_model(img)
    #         if (i + 1) % 100 == 0:
    #             print(f"  Calibrated {batch_size * (i + 1)} images...")


    # print("calibrated:", prepared_model)

    # ---------- 转换为量化模型 ----------
    print("🔁 Converting quantized model...")
    quantized_model = convert_pt2e(prepared_model)

    quantized_model.to("cpu")

    print("quantized:", quantized_model)
    #
    # quantized_model.to(device)
    #
    # # ---------- 评估 ----------
    # print("🧪 Evaluating quantized model...")
    # criterion = nn.CrossEntropyLoss()
    # top1, top5 = evaluate(quantized_model, criterion, val_loader, max_batches=eval_batches)
    # print(f"✅ Eval done | top-1 {top1.avg:.2f}% | top-5 {top5.avg:.2f}%")
    #
    # # ---------- 保存模型 ----------
    # torch.save(quantized_model.state_dict(), pt_dir / "resnet18_static_quantized.pth")
    # print("💾 Quantized model saved to:", pt_dir / "resnet18_static_quantized.pth")

if __name__ == "__main__":
    main()

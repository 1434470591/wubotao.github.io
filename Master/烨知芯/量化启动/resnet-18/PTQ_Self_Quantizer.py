import os, time, math
import torch
import torch.nn as nn
from torch.ao.quantization import PerChannelMinMaxObserver, HistogramObserver, MinMaxObserver
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer import Quantizer, QuantizationSpec, QuantizationAnnotation, SharedQuantizationSpec
from torch.export import export
from torch.fx import GraphModule
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# =============== 超参数 ===============
DATA_DIR = "./data"
BATCH_SIZE = 128
NUM_WORKERS = min(8, os.cpu_count() or 1)
EPOCHS = 0                 # 设为 0 可跳过微调，直接做 PTQ
LR = 1e-3
CALIB_BATCHES = None       # 设为 N 可只用前 N 个 batch 做校准；None = 全量训练集
DEVICE_CALIB = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TRAIN = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_EVAL = "cpu"        # 目标为 CPU，导出/量化和评测均在 CPU

torch.manual_seed(0)

# =============== 兼容导出与量化 API ===============
USE_TORCHAO = False
USE_EXECUTORCH = False

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
try:
    _eval_tf = weights.transforms() if weights is not None else None
    norm = _eval_tf.transforms[-1]
    mean = getattr(norm, 'mean', IMAGENET_MEAN)
    std  = getattr(norm, 'std',  IMAGENET_STD)
except Exception:
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

pin_for_gpu = (DEVICE_CALIB == "cuda")
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=pin_for_gpu)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

aten = torch.ops.aten

def _is_conv2d(n):  return n.op == "call_function" and n.target == aten.conv2d.default
def _is_linear(n):  return n.op == "call_function" and n.target == aten.linear.default
def _is_relu(n):    return n.op == "call_function" and n.target in (aten.relu_.default, aten.relu.default)
def _is_maxpool(n): return n.op == "call_function" and n.target in (aten.max_pool2d.default, aten.max_pool2d_with_indices.default)
def _is_adapool(n): return n.op == "call_function" and n.target == aten.adaptive_avg_pool2d.default
def _is_flatten(n): return n.op == "call_function" and n.target == aten.flatten.using_ints
def _is_add(n):     return n.op == "call_function" and n.target in (aten.add.Tensor, aten.add_.Tensor)

# =============== 自定义 Quantizer ===============
class MyQuantizer(Quantizer):
    def __init__(self, per_channel_weight: bool = True):
        super().__init__()
        # --- 激活（uint8/affine/Histogram） ---
        self.act_qspec = QuantizationSpec(
            dtype=torch.uint8,
            quant_min=0, quant_max=255,
            qscheme=torch.per_tensor_affine,
            is_dynamic=False,
            observer_or_fake_quant_ctr=HistogramObserver.with_args(bins=1024),
        )
        # --- 权重（int8/对称/MinMax） ---
        if per_channel_weight:
            self.w_qspec = QuantizationSpec(
                dtype=torch.int8,
                quant_min=-128, quant_max=127,  # 对称范围
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
                is_dynamic=False,
                observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(reduce_range=False),
            )
        else:
            self.w_qspec = QuantizationSpec(
                dtype=torch.int8,
                quant_min=-128, quant_max=127,  # 对称范围
                qscheme=torch.per_tensor_symmetric,
                is_dynamic=False,
                observer_or_fake_quant_ctr=MinMaxObserver.with_args(reduce_range=False),
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
                # 权重量化
                if isinstance(n.args[1], torch.fx.Node):
                    wqa = QuantizationAnnotation(
                        input_qspec_map={},
                        output_qspec=self.w_qspec,
                        _annotated=True,
                    )
                    n.args[1].meta["quantization_annotation"] = wqa

            elif _is_relu(n) or _is_maxpool(n):
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

            elif _is_adapool(n) or _is_flatten(n):
                # 简化处理：不给共享 qparams，直接按普通激活量化
                qa = QuantizationAnnotation(
                    input_qspec_map={},
                    output_qspec=self.act_qspec,
                    _annotated=True,
                )
                n.meta["quantization_annotation"] = qa

        return gm

    # ---- validate：简单覆盖率检查（帮助发现漏标）----
    def validate(self, gm: GraphModule) -> None:
        must_cover = []
        for n in gm.graph.nodes:
            if _is_conv2d(n) or _is_linear(n) or _is_relu(n) or _is_maxpool(n) or _is_add(n) or _is_adapool(n) or _is_flatten(n):
                must_cover.append(n)
        missed = [n for n in must_cover if "quantization_annotation" not in n.meta]
        if missed:
            raise RuntimeError(f"Missed nodes: {[str(m.target) for m in missed]}")

class MyQuantizerWOnly(Quantizer):
    """
    只量化权重的 Quantizer：
    - conv/linear 的权重使用 MinMax（int8，对称），默认 per-channel（ch_axis=0）
    - 不为任何激活插 observer，也不设置 output_qspec
    """

    def __init__(self, per_channel_weight: bool = True):
        super().__init__()
        if per_channel_weight:
            self.w_qspec = QuantizationSpec(
                dtype=torch.int8,
                quant_min=-128, quant_max=127,
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
                is_dynamic=False,
                observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(reduce_range=False),
            )
        else:
            self.w_qspec = QuantizationSpec(
                dtype=torch.int8,
                quant_min=-128, quant_max=127,
                qscheme=torch.per_tensor_symmetric,
                is_dynamic=False,
                observer_or_fake_quant_ctr=MinMaxObserver.with_args(reduce_range=False),
            )

    def annotate(self, gm: GraphModule) -> GraphModule:
        for n in gm.graph.nodes:
            if _is_conv2d(n) or _is_linear(n):
                # 仅给“权重输入”标注量化规格；不设置 output_qspec、不插激活观察者
                qa = QuantizationAnnotation(
                    input_qspec_map={},
                    output_qspec=None,  # 关键：不量化激活
                    _annotated=True,
                )
                # n.args[1] 一般是 weight 的 get_attr Node
                if isinstance(n.args[1], torch.fx.Node):
                    qa.input_qspec_map[n.args[1]] = self.w_qspec
                n.meta["quantization_annotation"] = qa
        return gm

    def validate(self, gm: GraphModule) -> None:
        missing_w = []
        unexpected_act = []

        for n in gm.graph.nodes:
            if _is_conv2d(n) or _is_linear(n):
                qa = n.meta.get("quantization_annotation", None)
                w_node = n.args[1] if isinstance(n.args[1], torch.fx.Node) else None
                has_w = (
                    qa is not None
                    and hasattr(qa, "input_qspec_map")
                    and w_node in qa.input_qspec_map
                )
                if not has_w:
                    missing_w.append(n)

                # 确认没有激活量化（不应设置 output_qspec，也不应给输入激活标注）
                if qa is not None:
                    if getattr(qa, "output_qspec", None) is not None:
                        unexpected_act.append(n)
                    # 如果给了除 weight 之外的其它输入 qspec，也视为激活量化泄漏
                    extra_inputs = [
                        k for k in getattr(qa, "input_qspec_map", {}).keys()
                        if k is not (w_node)
                    ]
                    if len(extra_inputs) > 0:
                        unexpected_act.append(n)
            else:
                # 其它算子不应被标注
                if "quantization_annotation" in n.meta:
                    unexpected_act.append(n)

        if missing_w or unexpected_act:
            msgs = []
            if missing_w:
                msgs.append(f"Missing weight qspec on: {[str(m.target) for m in missing_w]}")
            if unexpected_act:
                msgs.append(f"Unexpected activation annotations on: {[str(u.target) for u in unexpected_act]}")
            raise RuntimeError(" ; ".join(msgs))

class MyQuantizerActOnly(Quantizer):
    def __init__(self, hist_bins: int = 512):
        super().__init__()
        self.act_qspec = QuantizationSpec(
            dtype=torch.uint8,
            quant_min=0, quant_max=255,
            qscheme=torch.per_tensor_affine,
            is_dynamic=False,
            observer_or_fake_quant_ctr=HistogramObserver.with_args(bins=hist_bins),
        )

        from torch.ao.quantization.quantizer import SharedQuantizationSpec as _SQS
        self.SharedQuantizationSpec = _SQS

    def annotate(self, gm: GraphModule) -> GraphModule:
        for n in gm.graph.nodes:
            if _is_conv2d(n) or _is_linear(n):
                # 只量化激活：输入/输出插观察者；不标注权重
                x = n.args[0]
                qa = QuantizationAnnotation(
                    input_qspec_map={x: self.act_qspec},
                    output_qspec=self.act_qspec,
                    _annotated=True,
                )
                n.meta["quantization_annotation"] = qa

            elif _is_relu(n) or _is_maxpool(n) or _is_adapool(n) or _is_flatten(n):
                qa = QuantizationAnnotation(
                    input_qspec_map={},
                    output_qspec=self.act_qspec,
                    _annotated=True,
                )
                n.meta["quantization_annotation"] = qa

            elif _is_add(n):
                a, b = n.args[0], n.args[1]
                if self.SharedQuantizationSpec is not None and isinstance(a, torch.fx.Node):
                    # 关键：用 (a, n) 这条“输入边”作为共享基准
                    share = self.SharedQuantizationSpec((a, n))
                    qa = QuantizationAnnotation(
                        input_qspec_map={a: self.act_qspec, b: share},
                        output_qspec=share,
                        _annotated=True,
                    )
                else:
                    qa = QuantizationAnnotation(
                        input_qspec_map={a: self.act_qspec, b: self.act_qspec},
                        output_qspec=self.act_qspec,
                        _annotated=True,
                    )
                n.meta["quantization_annotation"] = qa
        return gm

    def validate(self, gm: GraphModule) -> None:
        must_cover, weight_nodes = [], []
        for n in gm.graph.nodes:
            if _is_conv2d(n) or _is_linear(n) or _is_relu(n) or _is_maxpool(n) or _is_add(n) or _is_adapool(n) or _is_flatten(n):
                must_cover.append(n)
            if (_is_conv2d(n) or _is_linear(n)) and isinstance(n.args[1], torch.fx.Node):
                weight_nodes.append(n.args[1])
        missed = [n for n in must_cover if "quantization_annotation" not in n.meta]
        if missed:
            raise RuntimeError(f"Missed nodes (no activation quant): {[str(m.target) for m in missed]}")
        bad = [w for w in weight_nodes if "quantization_annotation" in w.meta]
        if bad:
            raise RuntimeError(f"Unexpected weight quantization found on nodes: {[str(b.target) for b in bad]}")

# =============== 模型构建 ===============
def build_model(num_classes=10):
    m = resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

@torch.no_grad()
def evaluate(model, loader, device="cpu", ir_eval=False):
    if ir_eval:
        torch.ao.quantization.move_exported_model_to_eval(model)
    else:
        model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred.cpu() == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100.0

# =============== 仅微调最后一层 ===============
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
    # torch.export 期望 example args 是一个 *tuple*，统一封装
    if isinstance(example_inputs, torch.Tensor):
        ex_args = (example_inputs,)
    elif isinstance(example_inputs, (list, tuple)):
        ex_args = tuple(example_inputs)
    else:
        raise TypeError(f"Unsupported example_inputs type: {type(example_inputs)}")

    exported = export(model, ex_args)
    # capture_pre_autograd_graph 分支需要 .module() 拿到 GraphModule
    return exported.module()

# =============== 量化（PTQ） ===============
def quantize_pt2e(exported_model, calib_loader):
    # 量化器配置：对称量化，按需可以切换 is_per_channel=True
    quantizer = MyQuantizer()

    prepared = prepare_pt2e(exported_model, quantizer)
    print("prepared:", prepared)
    # 校准：在 GPU（若可用）上前向，收集观察者统计
    # 兼容不同版本的 eval 入口
    try:
        from torch.ao.quantization.quantize_pt2e import move_exported_model_to_eval
        move_exported_model_to_eval(prepared)
    except Exception:
        torch.ao.quantization.move_exported_model_to_eval(prepared)
    prepared.to(DEVICE_CALIB)

    seen = 0
    t0 = time.time()
    with torch.inference_mode():
        for i, (images, _) in enumerate(calib_loader):
            images = images.to(DEVICE_CALIB, non_blocking=True)
            _ = prepared(images)
            seen += 1
            if CALIB_BATCHES is not None and seen >= CALIB_BATCHES:
                break
    if DEVICE_CALIB == "cuda":
        torch.cuda.synchronize()
    print(f"[calib] device={DEVICE_CALIB}  batches={seen}  time={time.time()-t0:.2f}s")

    # 转回 CPU 做 convert 并在 CPU 上运行 int8
    prepared.to("cpu")
    quantized = convert_pt2e(prepared)
    print("quantized:", quantized)
    return quantized

# =============== 主流程 ===============
def main():
    # 1) 构建并微调
    float_model = build_model()
    float_model = quick_finetune(float_model, epochs=EPOCHS, device=DEVICE_TRAIN)
    base_acc = evaluate(float_model.to(DEVICE_EVAL), test_loader, device=DEVICE_EVAL)
    print(f"[fp32] Top-1 = {base_acc:.2f}%")

    # 2) 导出 FP32 图
    example = torch.randn(1, 3, 224, 224)
    exported_fp32 = export_for_pt2e(float_model, example)

    # 3) 量化
    quantized_model = quantize_pt2e(exported_fp32, train_loader)

    # 4) 评测量化模型（CPU）
    q_acc = evaluate(quantized_model, test_loader, device=DEVICE_EVAL, ir_eval=True)
    print(f"[int8] Top-1 = {q_acc:.2f}%")

    # 5) 可选：保存
    torch.save(float_model.state_dict(), "resnet18_cifar10_fp32.pth")
    torch.save(quantized_model.state_dict(), "resnet18_cifar10_pt2e_int8.pth")
    print("Saved: fp32 -> resnet18_cifar10_fp32.pth ; int8 -> resnet18_cifar10_pt2e_int8.pth")

if __name__ == "__main__":
    main()

"""Ternary weight quantization: STE forward with FOGZO backward + TernaryLinear."""
import torch
import torch.nn as nn
import torch.nn.functional as F

_TEMP_MIN = 0.01
_TEMP_MAX = 10.0


class TernaryQuantizeFunction(torch.autograd.Function):
    """Forward: per-tensor threshold ternarization to {-1, 0, +1}.
    Backward (FOGZO): grad_latent = grad_output * (1 - tanh(W/temp)^2)."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        threshold = weight.abs().mean()
        mask = weight.abs() > threshold
        ternary = torch.sign(weight) * mask.to(weight.dtype)
        ctx.save_for_backward(weight, temperature)
        return ternary

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, temperature = ctx.saved_tensors
        temp = temperature.clamp(min=_TEMP_MIN, max=_TEMP_MAX)
        tanh_val = torch.tanh(weight / temp)
        scale = 1.0 - tanh_val.pow(2)
        grad_weight = grad_output * scale
        # No loss gradient for temperature (forward does not depend on it).
        return grad_weight, None


class TernaryLinear(nn.Module):
    """Linear layer with a latent FP32 weight quantized to ternary at forward time."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)
        self.temperature = nn.Parameter(torch.ones(1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = TernaryQuantizeFunction.apply(self.weight, self.temperature)
        return F.linear(x, w.to(x.dtype), self.bias)


class TernaryLinearInference(nn.Module):
    """Post-training linear with pre-quantized int8 ternary weights (no STE)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_trained(cls, layer: TernaryLinear) -> "TernaryLinearInference":
        with torch.no_grad():
            w = TernaryQuantizeFunction.apply(layer.weight, layer.temperature)
        inst = cls(layer.in_features, layer.out_features, bias=layer.bias is not None)
        inst.weight_int8.copy_(w.to(torch.int8))
        if layer.bias is not None:
            inst.bias.data.copy_(layer.bias.data)
        return inst

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_int8.to(x.dtype), self.bias)

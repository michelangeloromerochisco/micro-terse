"""Export a trained TerseModel to GGUF (F16).

Writes every tensor as float16 into a GGUF container. This always produces a
readable file and is the path the completion auto-export uses — it never crashes
(the previous I2_S path raised because the `gguf` build lacks GGML type 36).

TARGET = llama.cpp. NOTE: a *servable* Terse GGUF still requires adding Terse
architecture support to llama.cpp itself — Terse-micro is ternary + MoE + MTP +
QK-norm, which no existing llama.cpp arch matches. That work (map module names to
llama.cpp canonical `blk.N.*` / MoE `ffn_*_exps` names, register the arch +
inference graph, and emit ternary weights as llama.cpp's TQ2_0 instead of F16)
is a separate effort done post-training from the saved checkpoints. Until then
this exporter guarantees a written F16 GGUF, not a runtime-loadable model.
"""
import torch

from terse.model.terse_model import TerseModel


def _add_meta(writer, cfg) -> None:
    writer.add_uint32("terse.block_count", cfg.num_layers)
    writer.add_uint32("terse.embedding_length", cfg.hidden_dim)
    writer.add_uint32("terse.attention.head_count", cfg.num_heads)
    writer.add_uint32("terse.attention.head_count_kv", cfg.num_kv_heads)
    writer.add_uint32("terse.vocab_size", cfg.vocab_size)


def export_gguf(model: TerseModel, out_path: str, arch: str = "bitnet-b1.58") -> str:
    """Write ``model`` to ``out_path`` as an all-F16 GGUF; returns the path."""
    import gguf

    from terse.model.ternary import TernaryLinear, TernaryQuantizeFunction

    # Store the *quantized* {-1,0,+1} weight a ternary linear actually uses at
    # inference (F16-exact, and re-quantization is idempotent on reload) rather
    # than the latent fp32 weight (whose F16 rounding can flip borderline signs).
    ternary: dict[str, torch.Tensor] = {}
    for mod_name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            with torch.no_grad():
                ternary[f"{mod_name}.weight"] = TernaryQuantizeFunction.apply(
                    module.weight, module.temperature
                )

    writer = gguf.GGUFWriter(out_path, arch)
    _add_meta(writer, model.config)

    for name, tensor in model.state_dict().items():
        tensor = ternary.get(name, tensor)
        writer.add_tensor(name, tensor.to(torch.float16).cpu().numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return out_path

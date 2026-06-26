"""Strip a training checkpoint to weights-only for off-pod eval.

Drops optimizer/scheduler/RNG state (~5 GB -> ~weights size) so a checkpoint can
be downloaded and evaluated off the training host with zero impact on the run.

    python scripts/strip_checkpoint.py checkpoints/step_36000.pt step_36000_weights.pt
"""
import sys

import torch


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    torch.save({"model": ckpt["model"], "step": ckpt.get("step")}, dst)
    print("wrote", dst)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# completion_watcher.sh — on CLEAN pretraining completion, run the full post-training
# pipeline, then stop the pod. Order is chosen so nothing valuable can be lost:
#   1. export the BASE GGUF (before ORPO)  -> pretrained model is safe no matter what
#   2. ORPO full fine-tune on the final checkpoint (verified bf16 + Adafactor config)
#   3. export the ORPO GGUF
#   4. stop the pod (end billing) — only AFTER artifacts are written
# A failed/crashed ORPO never destroys the base (step 1 already saved it); re-ORPO later
# is cheap. It acts ONLY on a clean supervisor exit; a crash/give-up is left alone.
set -uo pipefail
cd /workspace/micro-terse
POD_ID="${POD_ID:-fz40hjddp01n5k}"
LOG=logs/completion.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

# Export a GGUF from any checkpoint (stripped {"model"} OR full) via gguf_llamacpp.
export_gguf() {  # $1=checkpoint  $2=out.gguf
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 python -c "
import sys, torch, yaml
from terse.model.config import TerseConfig
from terse.model.terse_model import TerseModel
from terse.export.gguf_llamacpp import export_gguf_llamacpp
cfg = yaml.safe_load(open('configs/micro.yaml'))['model']
m = TerseModel(TerseConfig(**cfg))
st = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
m.load_state_dict(st['model'] if isinstance(st, dict) and 'model' in st else st)
print('wrote', export_gguf_llamacpp(m, sys.argv[2]))
" "$1" "$2" >> "$LOG" 2>&1
}

log "watcher started (pod $POD_ID)"
while true; do
  if pgrep -f 'scripts/supervisor.sh' >/dev/null || pgrep -f 'scripts/train.py' >/dev/null; then
    sleep 300; continue
  fi
  # Supervisor + training both gone. Only proceed on a CLEAN completion.
  if ! tail -n 25 logs/alerts.log | grep -q 'command exited cleanly'; then
    sleep 120; continue
  fi

  log "CLEAN PRETRAINING COMPLETION detected"
  CKPT="$(ls -t checkpoints/step_*.pt 2>/dev/null | head -1)"
  log "final checkpoint: ${CKPT:-NONE}"

  # 1. base GGUF first (safety artifact — pretrained model preserved regardless)
  log "exporting base GGUF -> terse-micro-base.gguf"
  export_gguf "$CKPT" /workspace/terse-micro-base.gguf && log "base GGUF OK" || log "BASE EXPORT FAILED"

  # 2. SFT (teach the chat template + identity) on the final pretrained checkpoint.
  #    Validated recipe (PC, 2026-06-18). ORPO then runs on the SFT'd model.
  SFTCKPT="$CKPT"
  if [ -f data/sft_corpus.jsonl ] && [ -n "$CKPT" ]; then
    log "running SFT (full-FT, bf16 + adamw, 3 epochs, grad-accum 16)"
    if python scripts/train_sft.py --checkpoint "$CKPT" \
        --config configs/micro.yaml --data data/sft_corpus.jsonl --device cuda \
        --dtype bf16 --optimizer adamw --lr 2e-5 --epochs 3 --grad-accum 16 \
        --warmup-ratio 0.03 --seq-len 1024 --save-every 1000 \
        --out /workspace/sft_final.pt >> "$LOG" 2>&1 && [ -f /workspace/sft_final.pt ]; then
      SFTCKPT=/workspace/sft_final.pt
      log "SFT done -> $SFTCKPT"
      export_gguf "$SFTCKPT" /workspace/terse-micro-sft.gguf \
        && log "SFT GGUF OK" || log "SFT GGUF FAILED"
    else
      log "SFT FAILED — base preserved; ORPO will run on the base checkpoint"
    fi
  else
    log "no data/sft_corpus.jsonl — skipping SFT; ORPO runs on the base checkpoint"
  fi

  # 3. ORPO full-FT (verified config) on the SFT'd model, then 4. ORPO GGUF
  if [ -f data/orpo_mix.jsonl ] && [ -n "$SFTCKPT" ]; then
    log "running ORPO (full-FT, bf16 + adafactor, 1 epoch) on $SFTCKPT"
    if python scripts/train_pref.py --method orpo --checkpoint "$SFTCKPT" \
        --config configs/micro.yaml --data data/orpo_mix.jsonl --device cuda \
        --dtype bf16 --optimizer adafactor --lr 1e-5 --lam 0.5 --epochs 1 \
        --seq-len 1024 --out /workspace/orpo_final.pt >> "$LOG" 2>&1; then
      log "ORPO done"
      if [ -f /workspace/orpo_final.pt ]; then
        log "exporting ORPO GGUF -> terse-micro-orpo.gguf"
        export_gguf /workspace/orpo_final.pt /workspace/terse-micro-orpo.gguf \
          && log "ORPO GGUF OK" || log "ORPO EXPORT FAILED"
      fi
    else
      log "ORPO FAILED — base model + base GGUF are preserved; re-run ORPO later"
    fi
  else
    log "missing orpo_mix.jsonl or checkpoint — skipping ORPO"
  fi

  # 4. stop the pod to end billing (artifacts already on /workspace, which persists)
  log "POST-TRAINING COMPLETE; stopping pod $POD_ID"
  runpodctl stop pod "$POD_ID" >> "$LOG" 2>&1 || log "STOP FAILED — stop the pod manually to end billing"
  break
done

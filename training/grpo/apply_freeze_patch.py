"""
Patch verl's fsdp_workers.py to freeze MoE expert weights.

gpt-oss-120b param names:
  FREEZE: model.layers.{i}.mlp.experts.{gate_up_proj, down_proj, *_bias}
  TRAIN:  model.layers.{i}.mlp.router.{weight, bias}
          model.layers.{i}.self_attn.*
          model.layers.{i}.*norm*
          model.embed_tokens, model.lm_head, etc.

Usage:
    python3.12 training/grpo/apply_freeze_patch.py          # apply
    python3.12 training/grpo/apply_freeze_patch.py --revert  # undo
"""
import argparse
import shutil
import os
import sys

FSDP_WORKERS = "/home/ssm-user/.local/lib/python3.12/site-packages/verl/workers/fsdp_workers.py"
BACKUP = FSDP_WORKERS + ".bak"

OLD = '            actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)'

NEW = '''            # === AIMO3: Freeze MoE expert weights, train dense only ===
            _frozen_n, _trainable_n = 0, 0
            for _pname, _param in actor_module_fsdp.named_parameters():
                # "experts" matches gate_up_proj, down_proj and their biases
                # "router" is NOT matched — stays trainable
                if "experts" in _pname:
                    _param.requires_grad = False
                    _frozen_n += _param.numel()
                else:
                    _trainable_n += _param.numel()
            if self.rank == 0:
                _total = _frozen_n + _trainable_n
                if _total > 0:
                    print(f"[AIMO3] Trainable: {_trainable_n/1e9:.2f}B ({100*_trainable_n/_total:.1f}%) | "
                          f"Frozen: {_frozen_n/1e9:.2f}B ({100*_frozen_n/_total:.1f}%)")
                for _i, (_n, _p) in enumerate(actor_module_fsdp.named_parameters()):
                    if _i < 8:
                        print(f"[AIMO3]   {'TRAIN' if _p.requires_grad else 'FROZE'} {_n}")
            _trainable_params = [p for p in actor_module_fsdp.parameters() if p.requires_grad]
            if self.rank == 0:
                print(f"[AIMO3] Optimizer receives {len(_trainable_params)} param tensors")
            actor_optimizer = build_optimizer(_trainable_params, optim_config)
            # === END AIMO3 ==='''


def apply_patch():
    if not os.path.exists(FSDP_WORKERS):
        print(f"ERROR: {FSDP_WORKERS} not found")
        sys.exit(1)

    with open(FSDP_WORKERS, "r") as f:
        source = f.read()

    if "AIMO3" in source:
        print("Already patched.")
        return

    if OLD not in source:
        print("ERROR: Cannot find target line.")
        for i, line in enumerate(source.split('\n'), 1):
            if 'build_optimizer' in line:
                print(f"  Line {i}: {line!r}")
        sys.exit(1)

    if not os.path.exists(BACKUP):
        shutil.copy2(FSDP_WORKERS, BACKUP)
        print(f"Backup: {BACKUP}")

    patched = source.replace(OLD, NEW, 1)

    with open(FSDP_WORKERS, "w") as f:
        f.write(patched)

    print("Patch applied.")


def revert():
    if os.path.exists(BACKUP):
        shutil.copy2(BACKUP, FSDP_WORKERS)
        print("Reverted.")
    else:
        print("No backup found.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--revert", action="store_true")
    args = p.parse_args()
    revert() if args.revert else apply_patch()

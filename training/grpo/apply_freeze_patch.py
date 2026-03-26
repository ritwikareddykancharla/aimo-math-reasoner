"""
Patch verl's fsdp_workers.py to freeze MoE expert weights.

Replaces the single optimizer creation line with:
  1. Freeze all *.experts.* params (requires_grad=False)
  2. Pass only trainable params to optimizer

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

# Exact line from sed -n '649p' (with leading whitespace)
OLD = '            actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)'

NEW = '''            # === AIMO3: Freeze MoE expert weights, train dense only ===
            import re as _re
            _frozen_n, _trainable_n = 0, 0
            for _pname, _param in actor_module_fsdp.named_parameters():
                if _re.search(r"\\.experts\\.", _pname) and not _re.search(r"gate|router", _pname, _re.IGNORECASE):
                    _param.requires_grad = False
                    _frozen_n += _param.numel()
                else:
                    _trainable_n += _param.numel()
            if self.rank == 0 and (_frozen_n + _trainable_n) > 0:
                _total = _frozen_n + _trainable_n
                print(f"[AIMO3] Trainable: {_trainable_n/1e9:.2f}B ({100*_trainable_n/_total:.1f}%) | "
                      f"Frozen: {_frozen_n/1e9:.2f}B ({100*_frozen_n/_total:.1f}%)")
            _trainable_params = [p for p in actor_module_fsdp.parameters() if p.requires_grad]
            if self.rank == 0:
                print(f"[AIMO3] Optimizer receives {len(_trainable_params)} trainable param tensors")
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
        print("ERROR: Cannot find target line. File may have changed.")
        print(f"Looking for:\n  {OLD!r}")
        # Show nearby lines for debugging
        for i, line in enumerate(source.split('\n'), 1):
            if 'build_optimizer' in line:
                print(f"  Found line {i}: {line!r}")
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

"""
Patch verl fsdp_workers.py for 120B MoE dense-only training:
  PATCH 1: Freeze experts before FSDP wrap (line ~500)
  PATCH 2: Enable CPU offload for actor (line ~599)
  PATCH 3: Optimizer only gets trainable params (line ~649)

Also requires config: actor.fsdp_config.use_orig_params=true
(passed via train_grpo.py command line)

Usage:
    python3.12 training/grpo/apply_freeze_patch.py
    python3.12 training/grpo/apply_freeze_patch.py --revert
"""
import argparse
import shutil
import os
import sys

FSDP_WORKERS = "/home/ssm-user/.local/lib/python3.12/site-packages/verl/workers/fsdp_workers.py"
BACKUP = FSDP_WORKERS + ".bak"


def apply_patch():
    if not os.path.exists(FSDP_WORKERS):
        print(f"ERROR: {FSDP_WORKERS} not found"); sys.exit(1)

    source_file = BACKUP if os.path.exists(BACKUP) else FSDP_WORKERS
    with open(source_file, "r") as f:
        source = f.read()

    if "AIMO3" in source:
        print("ERROR: source already patched"); sys.exit(1)

    if not os.path.exists(BACKUP):
        shutil.copy2(FSDP_WORKERS, BACKUP)
        print(f"Backup: {BACKUP}")

    # ── PATCH 1: Freeze experts before FSDP wrap ──────────────
    OLD1 = '            actor_module.to(torch_dtype)'
    NEW1 = '''            actor_module.to(torch_dtype)

            # === AIMO3 PATCH 1: Freeze MoE experts before FSDP wrap ===
            if role == "actor":
                _frozen_n, _trainable_n = 0, 0
                for _pname, _param in actor_module.named_parameters():
                    if "experts" in _pname and "router" not in _pname and "gate" not in _pname:
                        _param.requires_grad = False
                        _frozen_n += _param.numel()
                    else:
                        _trainable_n += _param.numel()
                if self.rank == 0:
                    _total = _frozen_n + _trainable_n
                    if _total > 0:
                        print(f"[AIMO3] Trainable: {_trainable_n/1e9:.2f}B ({100*_trainable_n/_total:.1f}%) | "
                              f"Frozen: {_frozen_n/1e9:.2f}B ({100*_frozen_n/_total:.1f}%)")
                    for _i, (_n, _p) in enumerate(actor_module.named_parameters()):
                        if _i < 8:
                            print(f"[AIMO3]   {'TRAIN' if _p.requires_grad else 'FROZE'} {_n}")
                # Force use_orig_params so FSDP respects requires_grad per param
                self.use_orig_params = True
                if self.rank == 0:
                    print("[AIMO3] Forced use_orig_params=True for per-param freeze")
            # === END AIMO3 PATCH 1 ==='''

    if OLD1 not in source:
        print("ERROR: Cannot find PATCH 1 target"); sys.exit(1)
    source = source.replace(OLD1, NEW1, 1)
    print("  ✓ PATCH 1: Freeze experts + use_orig_params")

    # ── PATCH 2: Enable CPU offload for actor ─────────────────
    OLD2 = '        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)'
    NEW2 = '''        # === AIMO3 PATCH 2: CPU offload for actor (120B needs this) ===
        cpu_offload = CPUOffload(offload_params=True)
        # === END AIMO3 PATCH 2 ==='''

    if OLD2 not in source:
        print("ERROR: Cannot find PATCH 2 target"); sys.exit(1)
    source = source.replace(OLD2, NEW2, 1)
    print("  ✓ PATCH 2: Actor CPU offload enabled")

    # ── PATCH 3: Optimizer only gets trainable params ─────────
    OLD3 = '            actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)'
    NEW3 = '''            # === AIMO3 PATCH 3: Only trainable params to optimizer ===
            _trainable_params = [p for p in actor_module_fsdp.parameters() if p.requires_grad]
            _all_params = list(actor_module_fsdp.parameters())
            if self.rank == 0:
                print(f"[AIMO3] Optimizer: {len(_trainable_params)} trainable / {len(_all_params)} total param tensors")
            actor_optimizer = build_optimizer(_trainable_params, optim_config)
            # === END AIMO3 PATCH 3 ==='''

    if OLD3 not in source:
        print("ERROR: Cannot find PATCH 3 target"); sys.exit(1)
    source = source.replace(OLD3, NEW3, 1)
    print("  ✓ PATCH 3: Optimizer tracks trainable only")

    with open(FSDP_WORKERS, "w") as f:
        f.write(source)
    print("All patches applied.")


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

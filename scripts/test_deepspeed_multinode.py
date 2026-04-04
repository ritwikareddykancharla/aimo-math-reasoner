"""
Test DeepSpeed multi-node communication between 2x p5.48xlarge (16x H100).

Run from Node 1 (head) only:

    deepspeed \
        --hostfile /data/aimo-math-reasoner/training/sft/hostfile \
        --master_addr 172.31.110.230 \
        --master_port 29500 \
        --num_nodes 2 \
        --num_gpus 8 \
        scripts/test_deepspeed_multinode.py

What it tests:
  1. All 16 GPUs are visible and initialized
  2. NCCL all-reduce works across nodes
  3. Bandwidth is reasonable (EFA should give >10 GB/s)
  4. Every rank can talk to every other rank
"""

import os
import time
import socket

import torch
import torch.distributed as dist
import deepspeed


def main():
    # DeepSpeed sets these env vars when launched via `deepspeed` CLI
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # Init distributed
    deepspeed.init_distributed(dist_backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9

    print(f"[Rank {rank:2d}/{world_size}] {ip} | GPU {local_rank} | {gpu_name} | {gpu_mem:.0f} GB")
    dist.barrier()

    # ── Test 1: Basic all-reduce ──────────────────────────────────────────
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 1: Basic all-reduce (sum of ranks)")
        print(f"{'='*60}")

    tensor = torch.tensor([float(rank)], device=f"cuda:{local_rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))

    ok = abs(tensor.item() - expected) < 0.01
    if rank == 0:
        print(f"  Sum of all ranks: {tensor.item():.0f} (expected {expected})")
        print(f"  Result: {'PASS' if ok else 'FAIL'}")

    dist.barrier()

    # ── Test 2: Bandwidth test (large all-reduce) ────────────────────────
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 2: All-reduce bandwidth (256 MB tensor)")
        print(f"{'='*60}")

    # 256 MB of float32 = 64M elements
    size = 64 * 1024 * 1024
    data = torch.randn(size, device=f"cuda:{local_rank}")

    # Warmup
    for _ in range(3):
        dist.all_reduce(data)
    torch.cuda.synchronize()

    # Timed runs
    n_iters = 10
    start = time.time()
    for _ in range(n_iters):
        dist.all_reduce(data)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    bytes_per_iter = size * 4  # float32 = 4 bytes
    total_bytes = bytes_per_iter * n_iters
    bw_gbps = total_bytes / elapsed / 1e9

    if rank == 0:
        print(f"  {n_iters} iters of 256 MB all-reduce in {elapsed:.2f}s")
        print(f"  Throughput: {bw_gbps:.1f} GB/s")
        if bw_gbps > 10:
            print(f"  Result: PASS (EFA likely working)")
        elif bw_gbps > 1:
            print(f"  Result: WARN (low bandwidth — EFA may not be active)")
        else:
            print(f"  Result: FAIL (very low bandwidth — check NCCL/EFA config)")

    del data
    torch.cuda.empty_cache()
    dist.barrier()

    # ── Test 3: Cross-node point-to-point ────────────────────────────────
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 3: Cross-node point-to-point (rank 0 <-> rank 8)")
        print(f"{'='*60}")

    if world_size >= 16:
        # Rank 0 = Node 1 GPU 0, Rank 8 = Node 2 GPU 0
        buf = torch.zeros(1024, device=f"cuda:{local_rank}")
        if rank == 0:
            buf.fill_(42.0)
            dist.send(buf, dst=8)
            dist.recv(buf, src=8)
            ok = (buf[0].item() == 84.0)
            print(f"  Rank 0 sent 42 -> Rank 8, got back {buf[0].item():.0f} (expected 84)")
            print(f"  Result: {'PASS' if ok else 'FAIL'}")
        elif rank == 8:
            dist.recv(buf, src=0)
            buf *= 2
            dist.send(buf, dst=0)
    else:
        if rank == 0:
            print(f"  SKIP (need 16 GPUs, have {world_size})")

    dist.barrier()

    # ── Test 4: Broadcast from each node ─────────────────────────────────
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST 4: Broadcast from each node")
        print(f"{'='*60}")

    for src in [0, min(8, world_size - 1)]:
        data = torch.tensor([float(src * 100 + 7)], device=f"cuda:{local_rank}")
        if rank == src:
            pass  # already has the value
        else:
            data.zero_()
        dist.broadcast(data, src=src)
        expected_val = float(src * 100 + 7)
        ok = abs(data.item() - expected_val) < 0.01
        if rank == 0:
            print(f"  Broadcast from rank {src}: got {data.item():.0f} (expected {expected_val:.0f}) -> {'PASS' if ok else 'FAIL'}")

    dist.barrier()

    # ── Summary ──────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"ALL TESTS DONE — {world_size} GPUs across {world_size // 8} node(s)")
        print(f"{'='*60}\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

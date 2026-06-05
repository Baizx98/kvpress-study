import os
import time

import torch


def main() -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    free, total = torch.cuda.mem_get_info(device)
    target = int(free * 0.965)
    chunk = 512 * 1024 * 1024
    blocks = []
    allocated = 0

    print(
        "memory_test pid={} physical_device={} total={}MiB free={}MiB target={}MiB".format(
            os.getpid(),
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            total // 1024 // 1024,
            free // 1024 // 1024,
            target // 1024 // 1024,
        ),
        flush=True,
    )

    while allocated < target:
        nbytes = min(chunk, target - allocated)
        try:
            block = torch.empty(nbytes, dtype=torch.uint8, device=device)
            block.fill_(1)
            blocks.append(block)
            allocated += nbytes
            if len(blocks) % 8 == 0:
                print(f"allocated={allocated // 1024 // 1024}MiB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"oom_at={allocated // 1024 // 1024}MiB; holding current allocation", flush=True)
            torch.cuda.empty_cache()
            break

    print(f"holding={allocated // 1024 // 1024}MiB blocks={len(blocks)}", flush=True)

    # Keep moderate, periodic compute activity without saturating the GPU.
    a = torch.randn((2048, 2048), dtype=torch.float16, device=device)
    b = torch.randn((2048, 2048), dtype=torch.float16, device=device)
    while True:
        _ = a @ b
        torch.cuda.synchronize(device)
        print(f"memory_test heartbeat holding={allocated // 1024 // 1024}MiB time={int(time.time())}", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()

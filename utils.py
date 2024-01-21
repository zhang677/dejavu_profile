import torch
import torch.nn as nn

ACT2FN = {
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "swish": nn.SiLU(),
}


def benchmark(fn, input, Wgate, Wup, Wdown, idx, activation="swish", warmup=20, rep=80, quantiles=None, fast_flush=True, return_mode="mean"):
    # https://github.com/nox-410/tvm.tl/blob/tl/python/tvm/tl/utils.py#L144    
    fn(input, Wgate, Wup, Wdown, idx, activation)
    torch.cuda.synchronize()

    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn(input, Wgate, Wup, Wdown, idx, activation)
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

    # Warm-up
    for _ in range(n_warmup):
        fn(input, Wgate, Wup, Wdown, idx, activation)
    
    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        start_event[i].record()
        fn(input, Wgate, Wup, Wdown, idx, activation)
        end_event[i].record()
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float
    )
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()
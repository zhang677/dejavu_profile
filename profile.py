import torch
import argparse
import time
from utils import benchmark, ACT2FN
from dejavu import mistral_mlp_sparse

def mistral_mlp_torch(x, Wgate, Wup, Wdown, idx, activation="relu"):
    """
    x: (B, hidden_size)
    Wgate: (hidden_size, intermediate_size)
    Wup: (hidden_size, intermediate_size)
    Wdown: (intermediate_size, hidden_size)
    """
    act_fn = ACT2FN[activation]
    return torch.matmul(act_fn(torch.matmul(x, Wgate)) * torch.matmul(x, Wup), Wdown)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile mare upperbound')
    parser.add_argument('--compress', type=int, default=4)
    parser.add_argument('--filename', type=str, default="./perfs/dejavu.csv")
    parser.add_argument('--record', type=bool)
    args = parser.parse_args()

    hidden_size = 4096
    intermediate_size = 14336
    K = int(intermediate_size // args.compress)
    B = 1
    input = torch.randn((B, hidden_size), dtype=torch.float16).cuda()
    gate = torch.randn((hidden_size, K), dtype=torch.float16).cuda()
    gate_full = torch.randn((hidden_size, intermediate_size), dtype=torch.float16).cuda()
    gate_t = gate_full.T.contiguous()
    up = torch.randn((hidden_size, K), dtype=torch.float16).cuda()
    up_full = torch.randn((hidden_size, intermediate_size), dtype=torch.float16).cuda()
    up_t = up_full.T.contiguous()
    down = torch.randn((K, hidden_size), dtype=torch.float16).cuda()
    down_full = torch.randn((intermediate_size, hidden_size), dtype=torch.float16).cuda()
    perm = torch.randperm(intermediate_size)[:K]
    print(perm)
    idx, _ = torch.sort(perm)
    idx = idx.cuda()
    print(idx)
    hidden_act = "swish"
    time_dejavu = benchmark(mistral_mlp_sparse, input, gate_t, up_t, down_full, idx, hidden_act)
    time_dense_torch = benchmark(mistral_mlp_torch, input, gate, up, down, idx, hidden_act)
    with open(args.filename, "a") as f:
        if args.record:
            f.write("{},{:.2f},{:.2f}\n".format(K, time_dense_torch, time_dejavu))
        print("{},{:.2f},{:.2f}".format(K, time_dense_torch, time_dejavu))

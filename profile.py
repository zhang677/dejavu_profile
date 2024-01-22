import torch
import argparse
import time
from utils import benchmark, ACT2FN, benchmark_pred, Config, MistralSparseMLP, degrad_tensor
from dejavu import mistral_mlp_sparse, mistral_mlp_sparse_fuse

def mistral_mlp_pred_sparse(x, Wu, Wsv, Bsv, svd_activation, Wgate, Wup, Wdownt, activation, K):
    """
    x: (batch, d)
    Wu: (d, r)
    Wsv: (r, 3.5d)
    Bsv: (3.5d)
    Wgate: (3.5d, d)
    Wup: (3.5d, d)
    Wdownt: (3.5d, d)
    idx: (M)
    """
    svd_act_fn = ACT2FN[svd_activation]
    probs = svd_act_fn(torch.matmul(torch.matmul(x, Wu),Wsv) + Bsv)[0]
    _, topk_idcs = torch.topk(probs, k=K, dim=-1)
    idx, _ = torch.sort(topk_idcs)
    return mistral_mlp_sparse(x, Wgate, Wup, Wdownt, idx, activation)

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
    parser.add_argument('--rank', type=int, default=128)
    parser.add_argument('--compress', type=int, default=4)
    parser.add_argument('--filename', type=str, default="./perfs/dejavu.csv")
    parser.add_argument('--record', type=bool)
    args = parser.parse_args()

    hidden_size = 4096
    intermediate_size = 14336
    rank = args.rank
    K = int(intermediate_size // args.compress)
    B = 1
    hidden_act = "swish"
    svd_act = "relu"
    input = torch.randn((B, hidden_size), dtype=torch.float16).cuda()

    # gate_full = torch.randn((hidden_size, intermediate_size), dtype=torch.float16).cuda()
    # gate_t = gate_full.T.contiguous()
    # up_full = torch.randn((hidden_size, intermediate_size), dtype=torch.float16).cuda()
    # up_t = up_full.T.contiguous()
    # down_full = torch.randn((intermediate_size, hidden_size), dtype=torch.float16).cuda()
    # u_proj = torch.randn((hidden_size, rank), dtype=torch.float16).cuda()
    # sv_proj = torch.randn((rank, intermediate_size), dtype=torch.float16).cuda()
    # sv_bias = torch.randn((intermediate_size), dtype=torch.float16).cuda()

    config = Config(hidden_size, intermediate_size, rank, K, torch.float16, 'swish', 'relu')
    mlp = MistralSparseMLP(config).to(torch.float16).cuda()
    mlp(input)
    input = degrad_tensor(input)
    gate_t = degrad_tensor(mlp.gate_proj.weight)
    up_t = degrad_tensor(mlp.up_proj.weight)
    # down_full = degrad_tensor(mlp.down_proj.weight).T.contiguous()
    # u_proj = degrad_tensor(mlp.u_proj.weight).T.contiguous()
    # sv_proj = degrad_tensor(mlp.sv_proj.weight).T.contiguous()
    down_full = degrad_tensor(mlp.down_proj.weight).T
    u_proj = degrad_tensor(mlp.u_proj.weight).T
    sv_proj = degrad_tensor(mlp.sv_proj.weight).T
    sv_bias = degrad_tensor(mlp.sv_proj.bias)

    gate = torch.randn((hidden_size, K), dtype=torch.float16).cuda()
    up = torch.randn((hidden_size, K), dtype=torch.float16).cuda()
    down = torch.randn((K, hidden_size), dtype=torch.float16).cuda()    
    perm = torch.randperm(intermediate_size)[:K]
    idx, _ = torch.sort(perm)
    idx = idx.cuda()
    time.sleep(2)

    time_dense_torch = benchmark(mistral_mlp_torch, input, gate, up, down, idx, hidden_act)
    time_dejavu = benchmark(mistral_mlp_sparse_fuse, input, gate_t, up_t, down_full, idx, hidden_act)
    time_pred_dejavu = benchmark_pred(mistral_mlp_pred_sparse, input, u_proj, sv_proj, sv_bias, svd_act, gate_t, up_t, down_full, hidden_act, K)
    with open(args.filename, "a") as f:
        if args.record:
            f.write("{},{:.2f},{:.2f},{:.2f}\n".format(K, time_dense_torch, time_dejavu, time_pred_dejavu))
        print("{},{:.2f},{:.2f},{:.2f}".format(K, time_dense_torch, time_dejavu, time_pred_dejavu))

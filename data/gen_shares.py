import torch

def generate_2pc_shares(tensor, modulus=2**64):
    """Generates additive secret shares for a tensor."""
    share_0 = torch.randint(0, modulus, tensor.shape, dtype=torch.int64)
    share_1 = (tensor.to(torch.int64) - share_0) % modulus
    return share_0, share_1

if __name__ == "__main__":
    data = torch.randn(1, 3, 224, 224) * 1000
    s0, s1 = generate_2pc_shares(data)
    print("Generated 2PC shares.")

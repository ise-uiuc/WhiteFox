
def m(x1):
    v1 = torch.matmul(x1, W1) + b1
    v2 = torch.matmul(v1, W2) + b2
    v3 = torch.matmul(v2, W3) + b3
    v4 = torch.matmul(v3, W4) + b4
    v5 = torch.matmul(v4, W5) + b5
    v6 = v5.sum()
    return v6

# Initializing trainable parameters
torch.manual_seed(42)  # Setting the random seed helps with debug.

W1 = torch.randn(60, 20) * 0.1
b1 = torch.randn(20) * 0.1
W2 = torch.randn(20, 10) * 0.1
b2 = torch.randn(10) * 0.1
W3 = torch.randn(10, 8) * 0.1
b3 = torch.randn(8) * 0.1
W4 = torch.randn(8, 4) * 0.1
b4 = torch.randn(4) * 0.1
W5 = torch.randn(4)


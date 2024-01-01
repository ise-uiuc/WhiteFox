
def m1(x1):
    x2 = torch.nn.functional.dropout(x1, p=0.4)
    x3 = x2.reshape(size=(-1, 1))
    x4 = torch.rand_like(x3, dtype=None)
    return x4

# Input for the model
x1 = torch.randn(2, 10)

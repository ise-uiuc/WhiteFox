
def fn(x1):
    l1 = m(x1)
    l2 = l1 * torch.clamp(l1 + 3, min=0, max=6)
    l3 = l2 / 6
    return l3

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

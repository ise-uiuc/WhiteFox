
def fn(x):
    v1 = torch.nn.functional.interpolate(x, scale_factor=3.281095)
    v2 = v1 + 3
    v3 = v2.max(min=0)
    v4 = v3.max(max=6)
    v5 = v1 * v4
    v6 = v5 / 6
    return v6
# Inputs to the model
x1 = torch.randn(1, 4, 24, 24)

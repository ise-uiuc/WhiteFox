
def func(x1):
    v1 = x1.view(size=(-1, 7, 4, 5))
    v2 = v1 * 0.5
    v3 = v1 * v1 * v1
    v4 = v3 * 0.044715
    v5 = v1 + v4
    v6 = v5 * 0.7978845608028654
    v7 = torch.tanh(v6)
    v8 = v7 + 1
    v9 = v2 * v8
    return v9
# Inputs to the model
x1 = torch.randn(1, 7, 5, 4)
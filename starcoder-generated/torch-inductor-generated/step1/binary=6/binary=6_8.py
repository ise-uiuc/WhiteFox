
def fn(x):
    v1 = torch.nn.functional.linear(x, torch.rand(64, 64))
    return v1

# Initializing the model
v1 = torch.rand(64, 64)
v2 = torch.rand(64, 64)
m = lambda x: fn(x, other=v1) - v2

# Inputs to the model, x1 and x2 are two tensors of the same shape and type

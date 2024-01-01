
v1 = torch.randn(1, 3, 64, 64)
v2 = 3 + v1
v3 = v2.clamp(min=0, max=6)
v4 = v3.div(6)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 16, 32, 32)

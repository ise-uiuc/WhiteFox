
v1 = x1.permute(0, 1, 3, 2)
v2 = x2.permute(0, 1, 3, 2)
v3 = v1.permute(0, 1, 3, 2)
v4 = v1.permute(0, 1, 3, 2)
v5 = v2.permute(0, 1, 3, 2)
v6 = v3.permute(0, 1, 3, 2)
v7 = v4.permute(0, 1, 3, 2)
v8 = v5.permute(0, 1, 3, 2)
Model(torch.nn.Module)
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
x2 = torch.randn(1, 2, 2, 2)

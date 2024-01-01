
x2 = torch.randn(1, 3, 1, 64)
x3 = F.interpolate(x2, (3, 128))
x4 = torch.softmax(x3)
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

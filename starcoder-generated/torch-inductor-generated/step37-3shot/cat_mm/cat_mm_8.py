
x = torch.randn(8, 8)
for loopVar1 in range(100):
    x = torch.mm(x, x)
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)

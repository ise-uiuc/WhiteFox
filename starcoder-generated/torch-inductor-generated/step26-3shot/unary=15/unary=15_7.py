
x1 = torch.randn(4, 3, 10, 10)
model = torch.jit.trace(Model(), x1)
print(model.__repr__())

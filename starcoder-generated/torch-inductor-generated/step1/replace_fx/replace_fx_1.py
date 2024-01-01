 and inputs
x1 = torch.randn(1, 32, 32)
x2 = torch.randn(1, 32, 32)
m = CustomModule(x1)

y = m(x2)

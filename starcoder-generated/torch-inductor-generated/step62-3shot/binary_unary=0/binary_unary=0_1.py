
t1 = torch.randn((1, 16, 14, 14))
t2 = t1 + torch.randn((1, 16, 14, 14))
t3 = t2 + torch.tensor([16.9969])
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)

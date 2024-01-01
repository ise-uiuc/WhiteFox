
seq = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(1), torch.nn.Conv2d(3, 3, 3), torch.nn.BatchNorm2d(1))
# Inputs to the model
x2 = torch.randn(1, 3, 3, 3)

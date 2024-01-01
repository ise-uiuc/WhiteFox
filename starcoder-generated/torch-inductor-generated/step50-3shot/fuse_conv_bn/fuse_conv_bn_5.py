
model = torch.nn.Sequential(torch.nn.Conv2d(8, 8, 1), torch.nn.BatchNorm2d(8))
# Inputs to the model
x = torch.randn(1, 8, 16, 16)


model = nn.Sequential(torch.nn.Conv2d(2, 4, 3), torch.nn.BatchNorm2d(4))
torch.manual_seed(3)
model[0].weight = torch.nn.Parameter(torch.randn(model[0].weight.shape))
model[0].bias = torch.nn.Parameter(torch.randn(model[0].bias.shape))
model[1].running_mean = torch.arange(4, dtype=torch.float)
model[1].running_var = torch.arange(4, dtype=torch.float) * 2 + 1
# Inputs to the model
x1 = torch.randn(4, 2, 4, 4)

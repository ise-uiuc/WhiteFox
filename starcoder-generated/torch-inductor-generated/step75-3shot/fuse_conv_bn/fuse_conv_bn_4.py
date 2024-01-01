
model = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 1, bias=False), torch.nn.BatchNorm2d(3, affine=False, momentum=0))
# Inputs to the model
x1 = torch.randn(1, 2, 3, 3)

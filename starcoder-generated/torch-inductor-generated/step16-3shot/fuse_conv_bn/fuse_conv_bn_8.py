
self.conv = nn.Conv2d(3, 3, kernel_size=3)
self.bn = nn.BatchNorm2d(3)
# Inputs to the model
x = torch.randn(1, 3, 4, 4)

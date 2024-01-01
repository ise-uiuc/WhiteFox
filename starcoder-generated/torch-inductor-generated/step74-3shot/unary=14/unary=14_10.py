
class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        modules = [torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                   torch.nn.BatchNorm2d(out_channels), torch.nn.ReLU(inplace=True)]
        self.conv2d = torch.nn.Sequential(*modules)
    def forward(self, input):
        return self.conv2d(input)
class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_14 = torch.nn.Sequential(torch.nn.Linear(512, 4096), torch.nn.BatchNorm1d(4096), torch.nn.ReLU())
    def forward(self, x1):
        v1 = self.linear_14(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.abs(v2)
        return v3
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.modules = [Conv2d(64, 64, 4, stride=2, padding=1), Conv2d(64, 64, 3, stride=1, padding=1),
                        Conv2d(64, 64, 1, stride=1, padding=0),
                        Linear()]
    def forward(self, x1):
        x11 = self.modules[0](x1)
        x12 = self.modules[1](x11)
        x13 = self.modules[2](x12)
        x14 = self.modules[3](x13)
        return x14
# Inputs to the model
x1 = torch.randn(1, 64, 284, 284)

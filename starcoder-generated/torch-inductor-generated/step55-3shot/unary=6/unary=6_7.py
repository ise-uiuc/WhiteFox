
class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False)
            self.bn = torch.nn.BatchNorm2d(1)
            self.relu = torch.nn.ReLU(inplace=False)

        def forward(self, x):
            x_ = self.conv(x)
            x = self.bn(x_)
            x = self.relu(x)
            return x
# Inputs to the model
x = torch.randn(1, 1, 16, 16)

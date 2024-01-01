
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=7, groups=3, padding=3)
        self.conv1 = torch.nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes * block.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.se_module = SELayer(planes * block.expansion, reduction=se_reduction)
    def forward(self, x):
        residual = x
        output = self.conv(x)
        output = self.bn1(output)
        output = self.relu(output)
        return output
# Inputs to the model
x1 = torch.randn(32, 3, 26, 26)

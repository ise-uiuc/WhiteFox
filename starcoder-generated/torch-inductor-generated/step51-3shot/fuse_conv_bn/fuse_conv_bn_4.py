
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_bn_relu = torch.nn.Sequential(torch.nn.Conv2d(12, 16, 5), torch.nn.BatchNorm2d(16), torch.nn.ReLU(inplace=True))
        self.conv_relu = torch.nn.Sequential(torch.nn.Conv2d(6, 16, 5), torch.nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.conv_relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 12, 5, 5)

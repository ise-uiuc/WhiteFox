
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(256, 256, 1, stride=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(256)
        self.relu1 = torch.nn.ReLU
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = self.bn(x1)
        x3 = self.relu1(x2)
        return x3
# Inputs to the model
x = torch.randn(32, 256, 14, 14)

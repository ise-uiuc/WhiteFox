
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unpool = torch.nn.Unpool2d(2)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 2, stride=2, padding=0)
        self.conv = torch.nn.Conv2d(3, 3, 2, stride=2, padding=1, dilation=2)
        self.tanh = torch.nn.Tanh()
        self.avgpool = torch.nn.AvgPool2d(2)
    def forward(self, x1):
        v1 = self.unpool(x1)
        v2 = self.conv_transpose(v1)
        v3 = self.conv(x1)
        v4 = self.tanh(v2)
        v5 = self.avgpool(v4)
        v6 = self.conv_transpose(v5)
        v7 = self.conv_transpose(v3)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)

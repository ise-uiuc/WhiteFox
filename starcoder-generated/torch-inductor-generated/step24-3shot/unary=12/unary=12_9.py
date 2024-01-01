
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 512, 1, stride=1, padding=1)
        self.conv_1 = torch.nn.Conv2d(256, 384, 1, stride=3, padding=1)
        self.conv_2 = torch.nn.Conv2d(64, 128, 1, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.sigmoid()
        v4 = self.conv_1(v2)
        v5 = v4 * v2
        v6 = v5.tanh()
        v7 = v6.sigmoid()
        v8 = v1 * v7
        return v8
# Inputs to the model
x1 = torch.randn(1, 64, 32, 32)

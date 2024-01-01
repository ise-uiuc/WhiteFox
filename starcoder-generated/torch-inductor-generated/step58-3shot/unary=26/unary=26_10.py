
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 8, 7, padding=1)
        self.conv = torch.nn.Conv2d(8, 3, 7, padding=1)
        self.relu6 = torch.nn.ReLU6()
        self.threshold = torch.nn.Threshold(0, 0.5)
    def forward(self, x1):
        v2 = self.conv_t(x1)
        v3 = v2 > 0
        v4 = self.conv(v2)
        v5 = self.relu6(v4)
        v6 = self.threshold(v5)
        v7 = v5.detach()
        v8 = v6.detach()
        v9 = torch.where(v3, v4, v8)
        v10 = torch.where(v3, v6, v7)
        return v10
# Inputs to the model
x1 = torch.randn(4, 1, 10, 31)

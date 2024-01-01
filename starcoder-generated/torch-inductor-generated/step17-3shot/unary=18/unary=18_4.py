
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.sigmoid = torch.nn.Sigmoid()
        self.sigmoid = torch.nn.Tanh()
        self.conv = torch.nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv = torch.nn.Conv2d(in_channels=384, out_channels=80, kernel_size=1, stride=1, padding=0)
        self.conv = torch.nn.Conv2d(in_channels=384, out_channels=40, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = self.conv(x1)
        v4 = self.sigmoid(v3)
        v5 = self.conv(x1)
        v6 = self.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 384, 192, 192)

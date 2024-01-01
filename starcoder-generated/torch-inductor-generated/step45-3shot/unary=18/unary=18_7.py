
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=2)
        self.t2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1, dilation=1)
        self.t3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1, dilation=1)
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.t2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.t3(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

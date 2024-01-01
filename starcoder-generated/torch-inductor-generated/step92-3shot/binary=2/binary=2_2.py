
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 256, 1, stride=1, padding=1)
        self.act = torch.nn.ReLU6()
        self.avgPool = torch.nn.AvgPool2d((1, 1), (1, 1), (0, 0), ceil_mode=False)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = self.act(v1)
        v3 = self.avgPool(v2).flatten(start_dim=1)
        v4 = v3 - 0.5
        return v4
# Inputs to the model
x3 = torch.randn(2, 3, 400, 16)

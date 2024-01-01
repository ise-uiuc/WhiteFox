
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 12, 1, stride=1, padding=3)
    def forward(self, x40):
        v1 = self.conv(x40)
        v2 = v1
        v3 = v1
        v4 = v3
        v5 = v4
        v6 = v5
        v7 = v6
        v8 = v7
        v9 = v8
        v10 = v9
        return v10
# Inputs to the model
x40 = torch.randn(1, 7, 18, 3)

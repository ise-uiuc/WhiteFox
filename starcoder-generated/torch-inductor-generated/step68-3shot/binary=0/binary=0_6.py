
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 1, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1.relu()
        v3 = v2.transpose(0, 1)
        v4 = v3.view(1, 1, 1)
        v5 = v4 + 1.0
        v6 = v5 + x2
        return v6
# Inputs to the model
x1 = torch.randn(1, 4, 16, 16)
x2 = 1

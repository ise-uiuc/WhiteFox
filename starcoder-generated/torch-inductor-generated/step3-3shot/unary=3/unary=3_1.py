
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = x1
        v2 = self.conv(v1)
        v3 = v2 + 1
        v4 = torch.sin(v3)
        v5 = v4 * 0.5
        v6 = torch.mean(v5)
        v7 = torch.tan(v6)
        return v7   
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

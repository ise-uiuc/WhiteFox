
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 100
        v3 = v2.unsqueeze(0)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 56, 56)

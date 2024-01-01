
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.5
        v3 = torch.nn.ReLU(v2)
        v4 = torch.mean(v3, 3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 72, 72)

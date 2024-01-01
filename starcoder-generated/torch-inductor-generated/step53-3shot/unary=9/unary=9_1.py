
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v4 = torch.clip(self.conv(x1)+3, 0, 6)
        v5 = v4/6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

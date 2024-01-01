
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
    def forward(self, x1):
        v2 = x1 - 2.0
        v3 = self.conv(v2)
        v4 = F.relu(v3)
        v5 = v4 - 0.8
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)

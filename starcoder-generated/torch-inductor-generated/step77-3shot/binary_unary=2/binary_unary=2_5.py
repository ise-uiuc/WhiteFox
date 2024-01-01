
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = self.conv(v1)
        v3 = v2 - 0.01
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 240, 400)

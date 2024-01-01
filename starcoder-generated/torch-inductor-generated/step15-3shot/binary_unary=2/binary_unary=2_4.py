
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 7, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1.2
        v3 = F.relu(v2)
        v4 = v3 - 0.004
        v5 = F.relu(v4)
        v6 = torch.squeeze(v5, 0)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

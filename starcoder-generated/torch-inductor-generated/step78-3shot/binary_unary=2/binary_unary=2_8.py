
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 20, 11, stride=2, padding=1)
        self.mean = torch.mean
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.mean(v1, (0))
        v3 = v2 + 7.5
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 48, 48)

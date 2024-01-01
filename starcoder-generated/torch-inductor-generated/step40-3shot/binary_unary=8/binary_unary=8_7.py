
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 8, 1, stride=1)
    def forward(self, x1):
        v1 = torch.squeeze(x1, 0)
        v2 = torch.squeeze(x1, 0)
        v3 = self.conv(v1 + v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 64, 1, 1)
x2 = torch.randn(1, 64, 1, 1)

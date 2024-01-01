
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 10, stride=5, padding=5)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = v3[0]
        return v4
# Inputs to the model
x1 = torch.randn(2, 1, 16, 16)

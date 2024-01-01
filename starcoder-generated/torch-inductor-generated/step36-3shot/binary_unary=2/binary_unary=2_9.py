
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.01
        v3 = F.relu(v2) + 0.002
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

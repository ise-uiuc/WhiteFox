
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=0)
    def forward(self, x1, x2):
        v2 = self.conv(x2)
        v1 = x1 + v2
        v3 = torch.relu(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)

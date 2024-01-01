
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = v1.transpose(0, 1)
        v2 = v1 - 5.0
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.ones(4, 1)

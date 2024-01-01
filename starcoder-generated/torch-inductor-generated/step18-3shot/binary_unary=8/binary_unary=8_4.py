
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=0)
    def forward(self, x1):
        x2 = F.pad(x1, (1, 1, 1, 1), mode='reflect')
        v1 = self.conv(x2)
        v2 = self.conv(x2)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

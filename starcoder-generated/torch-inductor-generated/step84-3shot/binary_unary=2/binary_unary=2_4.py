
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 32, 7, stride=1, padding=0)
    def forward(self, x1):
        v0 = self.conv(x1)
        v1 = v0 - v0
        v2 = F.relu(v1)
        v3 = v2 + v0
        v4 = torch.squeeze(v3, 0)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 14, 14)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 100, stride=1, padding=0)
    def forward(self, x1):
        v0 = self.conv(x1)
        v1 = v0 - v0
        v2 = F.relu(v1)
        v3 = torch.squeeze(v2, 0)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)

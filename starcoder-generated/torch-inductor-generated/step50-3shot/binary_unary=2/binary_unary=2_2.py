
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 7, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.0
        v3 = F.relu(v2)
        v4 = torch.squeeze(v3, -1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 256, 256)

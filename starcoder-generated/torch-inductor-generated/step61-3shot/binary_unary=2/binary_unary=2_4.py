
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.0023
        v3 = F.relu(v2)
        v4 = torch.nn.functional.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 119, 119)

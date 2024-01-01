
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 16, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = v1 * v2
        v4 = F.max(v3, dim=1, keepdim=True)[0] - 0.1
        v5 = F.relu(v4)
        return v5
# Inputs to the model
v1 = torch.randn(1, 8, 80, 80)
v2 = v1 + 0.01

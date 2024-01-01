
class Conv2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 5, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.relu(v1) * torch.relu(v1)
        v4 = torch.nn.functional.interpolate(v3, scale_factor=None)
        v5 = torch.relu(v1) * 0.6
        v7 = torch.nn.functional.sigmoid(v4) * 123 - 456
        v10 = torch.tanh(v5 + v7) + 1
        v11 = torch.tanh(v2) + v10
        return v11
# Inputs to the model
x1 = torch.randn(1, 1, 223, 319)

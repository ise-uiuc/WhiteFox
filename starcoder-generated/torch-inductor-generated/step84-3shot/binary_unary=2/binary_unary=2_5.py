
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.8
        v3 = F.relu(v2)
        v4 = torch.squeeze(v3, 0)
        return torch.nn.functional.interpolate(v4, v4.shape[1:], mode='nearest')
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)

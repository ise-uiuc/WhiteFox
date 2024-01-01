
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = torch.nn.functional.interpolate(v3, scale_factor=0.01)
        return v4
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)

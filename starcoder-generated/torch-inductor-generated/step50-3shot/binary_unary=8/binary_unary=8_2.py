
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.interpolate(v1, scale_factor=0.5)
        v3 = self.conv1(x1)
        v4 = torch.cat((v2, v3), dim=1)
        v5 = torch.relu(v4)
        v6 = self.conv1(x1)
        v7 = torch.nn.functional.interpolate(v6, scale_factor=0.5)
        v8 = self.conv1(x1)
        v9 = torch.cat((v7, v8), dim=1)
        v10 = torch.relu(v9)
        return v5 + v10
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

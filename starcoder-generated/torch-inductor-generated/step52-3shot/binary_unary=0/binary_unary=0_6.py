
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 32, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.interpolate(v1, scale_factor=(1.0, 2.0), mode='nearest')
        v3 = v2 + x2
        v4 = torch.nn.functional.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.nn.functional.interpolate(v5, scale_factor=(1.0, 1.0), mode='nearest')
        v7 = v6 + x1
        v8 = torch.nn.functional.relu(v7)
        v9 = self.conv3(v8)
        v10 = torch.nn.functional.interpolate(x1, scale_factor=(1.0, 2.0), mode='nearest')
        v11 = torch.nn.functional.interpolate(x2, scale_factor=(1.0, 2.0), mode='nearest')
        v12 = v9 + v10 + v11
        v13 = torch.nn.functional.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)

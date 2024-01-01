
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv2d_2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + x
        v3 = torch.relu(v2)
        v4 = self.conv2d_2(v3)
        v5 = v4 + x
        v6 = torch.relu(v5)
        v7 = v6.view(10)
        return v7
# Inputs to the model
x = torch.randn(1, 16, 64, 64)

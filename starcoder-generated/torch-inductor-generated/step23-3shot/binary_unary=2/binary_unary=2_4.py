
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 8, padding=4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 2.0
        v3 = F.relu(v2)
        v4 = torch.squeeze(v3, 0)
        v5 = self.conv2(v4)
        v6 = v5 - 3.0
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

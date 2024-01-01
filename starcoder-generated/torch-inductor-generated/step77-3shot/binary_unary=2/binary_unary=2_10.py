
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 8, 1)
        self.conv2 = torch.nn.Conv2d(8, 7, 1)
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = self.conv(v1)
        v3 = self.conv2(v2)
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 7, 30, 40)

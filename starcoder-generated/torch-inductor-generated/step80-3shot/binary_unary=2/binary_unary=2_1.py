
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.2
        v3 = self.conv2(v2)
        v4 = v3 - 0.3
        v5 = F.relu(v3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 8, 16, 16)
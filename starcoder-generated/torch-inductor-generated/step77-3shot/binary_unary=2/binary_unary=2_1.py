
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 3)
        self.conv2 = torch.nn.Conv2d(5, 7, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = F.relu(v2)
        v4 = v3 - 0.15
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 8, 10)

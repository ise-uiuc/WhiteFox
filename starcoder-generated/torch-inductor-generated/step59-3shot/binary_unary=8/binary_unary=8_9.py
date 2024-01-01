
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 5, stride=1, padding=8)
        self.conv2 = torch.nn.Conv2d(1, 5, 5, stride=1, padding=8)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv1(x)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x = torch.randn(1, 1, 300, 450)

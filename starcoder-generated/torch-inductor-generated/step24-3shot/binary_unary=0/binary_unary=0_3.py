
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(55, 25, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(25, 12, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x = torch.randn(2, 55, 64, 64)

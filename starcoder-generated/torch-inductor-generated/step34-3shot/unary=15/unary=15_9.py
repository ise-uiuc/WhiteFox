
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=3, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 32, 7, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v1)
        v4 = torch.relu(v3)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 35, 35)

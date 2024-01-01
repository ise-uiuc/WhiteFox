
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 42, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v1)
        v4 = torch.relu(v1)
        return v1
# Inputs to the model
x1 = torch.randn(2, 3, 256, 256)

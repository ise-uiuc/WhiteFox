
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 3, 2, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 17, 17)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v1)
        v5 = torch.div(v4, v3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

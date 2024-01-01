
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(10, 16, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.relu(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 32, 32)

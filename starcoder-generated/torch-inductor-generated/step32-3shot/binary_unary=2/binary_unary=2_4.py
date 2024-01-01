
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 32, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 1, 5)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 10, 32, 32)

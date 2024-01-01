
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 5, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, stride=2, padding=1)
    def forward(self, x1):
        v1 = F.relu6(inputs)
        v2 = F.relu(self.conv1(v1))
        v3 = F.relu(self.conv2(v2))
        return v3
# Inputs to the model
x1 = torch.randn(16, 10, 56, 56)

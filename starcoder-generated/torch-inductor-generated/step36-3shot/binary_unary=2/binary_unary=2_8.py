
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 1, stride=4)
        self.conv2 = torch.nn.Conv2d(5, 10, 1, stride=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 3
        v4 = F.relu(v3)
        return v4.flatten(1)
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

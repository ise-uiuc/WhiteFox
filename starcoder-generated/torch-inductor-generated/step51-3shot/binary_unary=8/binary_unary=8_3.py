
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = v1 + v1
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

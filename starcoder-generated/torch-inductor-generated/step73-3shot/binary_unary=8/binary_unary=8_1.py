
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 17, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 4, 3, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv1(x)
        v4 = self.conv1(x)
        v5 = v1 + v2 + v3 + v4
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 5, 224, 224)

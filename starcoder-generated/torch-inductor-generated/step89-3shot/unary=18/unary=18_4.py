
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 1, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)

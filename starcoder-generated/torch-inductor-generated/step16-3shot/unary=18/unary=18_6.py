
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv1(x1))
        v2 = torch.relu(self.conv2(v1))
        v3 = self.conv3(v2)
        v4 = v3 + 0.5
        v5 = torch.tanh(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

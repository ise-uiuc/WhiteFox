
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 96, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(96, 3, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(96, 3, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x2)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v1)
        v7 = v3 + v4
        v8 = torch.relu(v7)
        v9 = v2 + 1
        v10 = torch.relu(v9)
        return x1 + v8
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224, requires_grad=True)

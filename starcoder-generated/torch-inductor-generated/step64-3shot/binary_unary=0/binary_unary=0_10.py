
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v6 = self.conv2(v1)
        v7 = self.conv3(v1)
        v8 = v6 + v7
        v9 = torch.relu(v8)
        v12 = self.conv4(v9)
        v13 = v12 + 1
        v14 = torch.relu(v13)
        return v14
# Inputs to the model
input = torch.randn(1, 16, 64, 64, requires_grad=True)

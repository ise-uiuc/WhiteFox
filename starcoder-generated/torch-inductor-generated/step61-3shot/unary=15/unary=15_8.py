
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(5, 10, 5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(10, 20, 5, stride=2, padding=2)
        self.conv4 = torch.nn.Conv2d(20, 48, 3, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(4, 3, 512, 512)

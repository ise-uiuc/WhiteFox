
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(6, 13, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(13, 20, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(20, 27, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(27, 1, 1, stride=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        v7 = self.conv4(v6)
        v8 = torch.tanh(v7)
        v9 = self.conv5(v8)
        return v9
# Inputs to the model
x = torch.randn(1, 1, 224, 224)

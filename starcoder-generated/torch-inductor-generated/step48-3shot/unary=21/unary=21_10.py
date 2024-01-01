
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 30, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(30, 50, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(50, 80, 1, stride=1)
        self.conv4 = torch.nn.Conv2d(80, 6, 1, stride=1)
    def forward(self, x):
        v1 = torch.tanh(self.conv1(x))
        v2 = torch.tanh(self.conv2(v1))
        v3 = torch.tanh(self.conv3(v2))
        v4 = torch.tanh(self.conv4(v3))
        return v4
# Inputs to the model
x = torch.randn(6, 6, 55, 33)

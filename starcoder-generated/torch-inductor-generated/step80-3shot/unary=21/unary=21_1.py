
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 1, padding=2, stride=1)
        self.conv2 = torch.nn.Conv2d(10, 10, 3, padding=2, dilation=2, stride=1)
        self.conv3 = torch.nn.Conv2d(10, 20, 5, padding=3, dilation=3, stride=2)
        self.conv4 = torch.nn.Conv2d(20, 30, 1, padding=3, dilation=0, stride=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v2 = torch.tanh(v2)
        v3 = self.conv3(v2)
        v3 = torch.tanh(v3)
        v4 = self.conv4(v3)
        v4 = torch.tanh(v4)
        v5 = torch.tanh(torch.tanh(torch.tanh(v4)))
        return v5
# Inputs to the model
x = torch.randn(2, 3, 32, 32)

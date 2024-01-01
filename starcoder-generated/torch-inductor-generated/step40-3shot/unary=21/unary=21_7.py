
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 7, stride=2, padding=3, dilation=1)
        self.conv3 = torch.nn.Conv2d(6, 6, 7, stride=2, padding=6, dilation=2)
        self.conv2 = torch.nn.Conv2d(6, 13, 5, stride=1, padding=2, dilation=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv3(v2)
        v4 = torch.tanh(v3)
        return self.conv2(v4)
# Inputs to the model
x = torch.randn(50, 3, 224, 224)

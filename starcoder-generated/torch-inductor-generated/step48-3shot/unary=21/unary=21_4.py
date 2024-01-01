
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 49, 3, stride=2, padding=2)
        self.conv1 = torch.nn.Conv2d(49, 49, 3, dilation=2, padding=2)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.conv2 = torch.nn.Conv2d(49, 1, 1, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv1(v1)
        v3 = self.softmax(v2)
        v4 = self.conv2(v3)
        v5 = torch.tanh(v4)
        return v5
# Inputs to the model
x = torch.randn(49, 7, 33, 98)

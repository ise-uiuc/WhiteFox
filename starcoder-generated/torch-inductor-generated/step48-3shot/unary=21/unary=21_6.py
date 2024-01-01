
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 10, 1, dilation=2, padding=2)
        self.conv2 = torch.nn.Conv2d(10, 16, 1, dilation=2, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 10, 1, dilation=2, padding=2)
        self.conv4 = torch.nn.Conv2d(10, 4, 1, dilation=2, padding=2)
        self.conv5 = torch.nn.Conv2d(4, 16, 1, dilation=2, padding=2)
        self.tanh = torch.nn.Tanh()
        self.conv6 = torch.nn.Conv2d(16, 16, 1, dilation=2, padding=2)
        self.linear = torch.nn.Linear(16, 10)
        self.softmax = torch.nn.LogSoftmax(dim=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.tanh(v5)
        v7 = self.conv6(v6)
        v8 = self.linear(v7)
        v9 = self.softmax(v8)
        return v9
# Inputs to the model
x = torch.randn(16, 3, 16, 16)

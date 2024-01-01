
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(49, 41, 1, stride=1)
        self.conv2 = torch.nn.Conv1d(41, 46, 1, stride=1)
        self.conv3 = torch.nn.Conv1d(46, 43, 1, stride=1)
        self.conv4 = torch.nn.Conv1d(43, 41, 1, stride=1)
        self.conv5 = torch.nn.Conv1d(41, 33, 1, stride=1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.tanh(x1)
        x3 = torch.tanh(x2)
        x4 = torch.tanh(x3)
        x5 = self.conv2(x4)
        x6 = torch.tanh(x5)
        x7 = self.conv3(x6)
        x8 = torch.tanh(x7)
        x9 = self.conv4(x8)
        x10 = torch.tanh(x9)
        x11 = self.conv5(x10)
        x12 = torch.tanh(x11)
        return x12
# Inputs to the model
x = torch.randn(1, 49, 54)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(4, 6, 4, stride=1, dilation=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(6, 2, 2, stride=1, dilation=1)
        self.relu2 = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu1(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 64)

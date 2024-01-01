
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 3, 16, stride=1)
        self.conv2 = torch.nn.Conv1d(3, 3, 16, stride=1)
        self.conv3 = torch.nn.Conv1d(3, 1, 3, stride=2)
        self.conv4 = torch.nn.Conv1d(1, 2, 5, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        return self.conv3(v4)
# Inputs to the model
x = torch.randn(1, 1, 4024)

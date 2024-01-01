
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 6, 20, stride=10, padding=5)
        self.conv2 = torch.nn.Conv1d(6, 12, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv1d(12, 20, 20, stride=10, padding=5)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        return torch.tanh(v4)
# Inputs to the model
x = torch.randn(15, 2, 120)


class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(16, 63, 1, stride=1)
        self.conv1d = torch.nn.Conv1d(16, 125, 1, stride=1)
    def forward(self, x0):
        v1 = self.conv2d(x0)
        v2 = torch.tanh(v1)
        v3 = self.conv1d(v2)
        return torch.tanh(v3)
x0 = torch.randn(32, 16, 199, 299)

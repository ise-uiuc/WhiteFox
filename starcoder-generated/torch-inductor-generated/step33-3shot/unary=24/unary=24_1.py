
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(16, 16, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv1d(x)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64)

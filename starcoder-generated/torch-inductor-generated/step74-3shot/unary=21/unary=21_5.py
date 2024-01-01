
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 64, 1, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.linear = torch.nn.Linear(64,1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn1(v1)
        v = v2.reshape(v2.shape[0], -1)
        v = self.linear(v)
        v = self.relu(v)
        v = torch.tanh(v)
        return v
# Inputs to the model
x = torch.randn(1, 2, 100, 100)

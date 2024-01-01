
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 4, 2, stride=2)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.relu(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 10, 33, 33)

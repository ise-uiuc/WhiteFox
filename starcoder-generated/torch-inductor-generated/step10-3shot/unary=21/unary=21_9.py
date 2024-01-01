
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 3)
        self.relu = torch.nn.ReLU6()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.relu(v1)
        return torch.tanh(v2)
# Inputs to the model
x = torch.randn(1, 2, 20, 20)

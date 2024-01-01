
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(168, 168, 1)
        self.relu = torch.nn.ReLU()
    def forward(self, x0):
        x1 = self.relu(self.conv(x0))
        x2 = torch.tanh(x1)
        return x2
# Inputs to the model
x0 = torch.randn(1, 168, 32, 32)


class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=2, padding=3)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 64, 64)


class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, kernel_size=5, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v = self.conv(x)
        v = torch.tanh(v)
        v = self.relu(v)
        return v
# Inputs to the model
x = torch.randn(1, 3, 224, 224)

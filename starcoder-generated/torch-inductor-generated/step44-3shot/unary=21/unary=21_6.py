
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ReflectionPad2d(2)
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=5)
    def forward(self, x):
        v1 = self.pad(x)
        v2 = self.relu(v1)
        v3 = self.conv(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 10, 10)

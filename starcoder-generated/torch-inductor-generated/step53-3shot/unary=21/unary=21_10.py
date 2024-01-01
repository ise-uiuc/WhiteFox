
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
    def forward(self, x):
        v1 = torch.tanh(x)
        v2 = self.conv(v1)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 19, 19)


class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(16, 8, 1)
    def forward(self, x):
        l0 = self.conv2d(x)
        v1 = torch.tanh(l0)
        return v1
# Inputs to the model
x = torch.randn(2, 16, 16, 16)

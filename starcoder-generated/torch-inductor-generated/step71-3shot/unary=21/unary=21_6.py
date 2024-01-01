
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxPool = torch.nn.MaxPool2d(13, stride=1)
    def forward(self, x):
        v1 = self.maxPool(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 32, 32)

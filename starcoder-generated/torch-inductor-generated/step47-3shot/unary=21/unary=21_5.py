
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 64, [1, 1])
    def forward(self, x):
        y = self.conv2d(x)
        out = torch.tanh(y)
        return out
# Inputs to the model
x = torch.randn(1, 3, 224, 224)

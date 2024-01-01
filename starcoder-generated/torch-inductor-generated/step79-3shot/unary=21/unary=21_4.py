
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = torch.nn.Conv2d(1, 3, 1)
    def forward(self, x):
        x = self.convolution(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 30, 46)

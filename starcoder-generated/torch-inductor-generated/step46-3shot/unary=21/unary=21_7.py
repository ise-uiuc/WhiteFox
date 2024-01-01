
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 3, 1, groups=1, bias=False)
    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = torch.tanh(x1)
        return x2
# Inputs to the model
x = torch.randn(1, 3, 10, 10)


class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False)
    def forward(self, x1):
        x2 = self.conv_1(x1)
        x3 = torch.tanh(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)

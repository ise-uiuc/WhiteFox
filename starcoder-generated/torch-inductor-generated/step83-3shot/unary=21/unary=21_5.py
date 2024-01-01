
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(4, 5, 4, 2, 0, bias=False, dilation=2)
    def forward(self, x):
        v1 = self.conv_1(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 4, 5, 5)

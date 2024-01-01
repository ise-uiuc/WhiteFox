
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 3, 3, groups=3, bias=True)
    def forward(self, x):
        v1 = self.conv_1(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 20, 32)

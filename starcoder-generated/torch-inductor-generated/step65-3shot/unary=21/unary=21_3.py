
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(6, 256, 1)
    def forward(self, x):
        v4 = self.conv_1(x)
        v5 = torch.sqrt(v4)
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 6, 10, 10)

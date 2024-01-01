
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(100, 150, 19, 3)
    def forward(self, x1):
        v1 = torch.tanh(self.conv(x1))
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 100, 170, 190)

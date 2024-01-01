
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 20, 64, stride=4)
    def forward(self, x):
        t1 = self.conv(x)
        t2 = torch.tanh(x)
        return t1 + t2
# Inputs to the model
x = torch.randn(1, 1, 64, 64)

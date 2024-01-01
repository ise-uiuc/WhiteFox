
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        h1 = self.conv(x)
        t1 = torch.tanh(h1)
        return t1
# Inputs to the model
x = torch.randn(1, 3, 76, 76)

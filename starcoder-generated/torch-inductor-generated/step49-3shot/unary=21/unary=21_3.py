
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3,16,3, padding=1, stride=1)
    def forward(self, x):
        y = self.conv(x)
        t = torch.tanh(y)
        return t
# Inputs to the model
x = torch.randn(1,3,48,48)

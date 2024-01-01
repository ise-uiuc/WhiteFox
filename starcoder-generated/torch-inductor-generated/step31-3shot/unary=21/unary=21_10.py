
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 128, 3, stride=1)
    def forward(self, x):
        t1 = self.conv(x)
        t2 = torch.tanh(t1)
        return t2
# Inputs to the model
x = torch.randn(10, 64, 21, 48)

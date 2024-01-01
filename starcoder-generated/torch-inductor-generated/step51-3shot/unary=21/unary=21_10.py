
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 128, 5, stride=1, padding=2)
    def forward(self, x):
        v1 = self.conv(x)
        t = torch.tanh(v1)
        return t
# Inputs to the model
x = torch.randn(1, 1, 112, 114)

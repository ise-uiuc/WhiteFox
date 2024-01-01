
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1, padding=0)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 109, 109)

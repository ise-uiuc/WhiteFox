
class myModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.my_conv = torch.nn.Conv2d(3, 48, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.my_conv(x)
        return torch.tanh(v1)
# Inputs to the model
x = torch.randn(1, 3, 64, 64)

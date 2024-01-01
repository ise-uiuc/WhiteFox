
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.Tanh()
        self.conv = torch.nn.Conv2d(2, 16, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.act(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)

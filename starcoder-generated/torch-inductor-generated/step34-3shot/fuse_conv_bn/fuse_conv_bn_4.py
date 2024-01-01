
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 3, 3)
        self.layernorm2d = torch.nn.LayerNorm([6, 6], 3e-3)
    def forward(self, x1):
        s = self.layernorm2d(x1)
        y = self.conv1(s)
        return s
# Inputs to the model
x1 = torch.randn(1, 2, 6, 6)

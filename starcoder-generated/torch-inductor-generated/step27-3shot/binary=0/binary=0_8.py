
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conva = torch.nn.Conv2d(1, 4, (1, 3), (1, 1), (0, 1))
        self.convb = torch.nn.Conv2d(4, 8, (7, 1), (1, 1), (0, 1))
    def forward(self, x1, other=1.0):
        var1 = self.conva(x1)
        var2 = self.convb(var1)
        var3 = var2 + other
        return var3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

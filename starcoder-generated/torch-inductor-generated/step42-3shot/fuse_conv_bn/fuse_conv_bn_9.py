
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv2d(2, 2, 3, padding=1)
        self.b = torch.nn.BatchNorm2d(2)
    def forward(self, x, l=True):
        if l:
            x = self.a(x)
            return x
        else:
            x = self.b(x)
            return x
# Inputs to the model
x = torch.randn(1, 2, 4, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 10, 1) 
        self.c2 = torch.nn.Conv2d(3, 10, 1)
        self.b = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        out_1 = self.c1(x)
        out_2 = self.c2(x)
        out = torch.cat([out_1, out_2], 1)
        out = self.b(out)
        return out
# Inputs to the model
x = torch.randn(1, 3, 1, 1)

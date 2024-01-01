
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 3, 3)
        self.c2 = torch.nn.Conv2d(3, 1, 1)
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return x
# Inputs to the model
x = torch.randn(3, 3, 10, 10)

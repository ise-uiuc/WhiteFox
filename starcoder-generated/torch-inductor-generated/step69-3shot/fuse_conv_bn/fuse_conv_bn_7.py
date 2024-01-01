
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 3, 3)
        self.b = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        b = self.b(x)
        c = torch.nn.functional.relu(self.c1(b))
        return c
# Inputs to the model
x = torch.randn(1, 3, 20, 20)

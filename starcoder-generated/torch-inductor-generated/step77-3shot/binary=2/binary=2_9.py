
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Parameter(torch.randn(1, 7, 5, 6, 8))
    def forward(self, x):
        t1 = self.conv.type_as(x)
        t2 = t1 + 1
        return t2
# Inputs to the model
x = torch.randn(2, 9, 32, 42)

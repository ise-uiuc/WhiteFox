
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        return torch.transpose(x1, 0, 1) + inp
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(1, 5)
inp = torch.randn(7, 8)

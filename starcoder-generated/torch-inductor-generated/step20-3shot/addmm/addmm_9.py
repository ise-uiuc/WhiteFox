
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.nn.functional.conv1d(x1, x2, stride=757)
        v2 = v1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, 39)
x2 = torch.randn(3, 39, 5)
inp = 17

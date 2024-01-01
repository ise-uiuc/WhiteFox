
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        x1 = x1.abs()
        return (v1 + inp).exp() + x2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3)
inp = torch.randn(1, 3)

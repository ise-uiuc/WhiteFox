
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = x1 + x2
        v2 = torch.mm(x1, inp)
        return torch.mm(v1, v2)
# Inputs to the model
x1 = torch.randn(2, 3, requires_grad=True)
x2 = torch.randn(2, 3)
inp = torch.nn.functional.relu(torch.rand(3, 3, requires_grad=True))

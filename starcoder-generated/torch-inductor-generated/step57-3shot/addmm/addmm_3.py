
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, inp):
        v1 = torch.mm(x, input)
        v2 = v1 + input
        v3 = torch.mm(v2, v2)
        return (v1, v3 + inp, torch.mm(v2.t(), v1))
# Inputs to the model
x = torch.randn(3, 3, requires_grad=True)
input = torch.randn(3, 3)

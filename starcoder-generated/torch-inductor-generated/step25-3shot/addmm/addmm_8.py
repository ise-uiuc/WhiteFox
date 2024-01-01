
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, inp):
        v1 = torch.mm(x, x)
        v2 = v1 + inp
        return v2.transpose(0, 1) 
# Inputs to the model
v1 = torch.randn(60, 10, requires_grad=True)
v2 = torch.randn(60, 5, requires_grad=True)
inp = torch.randn(10)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = torch.relu(v1 + x2)
        return v2
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5, requires_grad=True)
inp = torch.randn(5, 5)

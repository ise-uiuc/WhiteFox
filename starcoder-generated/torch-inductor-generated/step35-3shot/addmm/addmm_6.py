
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.mm(x, x)
        v2 = v1 + x
        s1 = torch.sum(v1)
        s2 = torch.sum(v2)
        return s1
# Inputs to the model
x = torch.randn(3, 3, requires_grad=True)

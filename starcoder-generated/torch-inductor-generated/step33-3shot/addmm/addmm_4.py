
# Solution
# Note: We do not have an input tensor 'inp' in this solution.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, *args):
        v1 = torch.mm(x1, x2)
        v2 = v1 + args[0]
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)

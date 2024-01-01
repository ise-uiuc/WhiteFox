
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.rand(3, 3, requires_grad=True))
        self.w2 = torch.nn.Parameter(torch.rand(3, 3, requires_grad=True))
        self.w3 = torch.nn.Parameter(torch.rand(3, 3, requires_grad=True))
        self.w4 = torch.nn.Parameter(torch.rand(3, 3, requires_grad=True))
        self.w5 = torch.nn.Parameter(torch.rand(3, 3, requires_grad=True))
        self.w6 = torch.nn.Parameter(torch.rand(3, 3, requires_grad=True))
        self.w7 = torch.nn.Parameter(torch.rand(3, 3, requires_grad=True))
    def forward(self, inp, x):
        v1 = torch.mm(inp, self.w3)
        v2 = v1 + x
        return v2
# Inputs to the model
x = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)

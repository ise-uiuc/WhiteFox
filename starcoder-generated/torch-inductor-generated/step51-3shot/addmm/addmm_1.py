
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, inp):
        v1 = torch.mm(inp, torch.randperm(inp.size()[0]).to(inp.dtype).to(inp.device))
        v2 = torch.mm(inp, x1)
        v3 = v2 * x2
        v4 = v1 + x3
        v5 = v4 + x4
        v6 = v5 + x5
        v7 = v3 + x6
        v8 = v7 * x9
        v9 = v6 + x5
        v10 = torch.mm(v9, torch.randperm(v9.size()[0]).to(v9.dtype).to(v9.device))
        v11 = v8 + v10
        v12 = torch.mm(v1, x8)
        y1 = v12 + x5
        y2 = v11 + x6
        return y2 + x7
# Inputs to the model
x1 = torch.zeros(3, 3)
x2 = torch.ones(3, 3)
x3 = torch.empty(3, 3)
x4 = torch.tensor(3)
x5 = torch.eye(3)
x6 = torch.randn(3, 3)
x7 = torch.empty(3, 3, requires_grad=True)
x8 = torch.ones(3, 3)
x9 = inp = torch.randn(3, 3)

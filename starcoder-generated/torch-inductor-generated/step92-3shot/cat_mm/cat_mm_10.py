
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x1)
        v2 = torch.mm(x2, x2)
        v3 = torch.mm(x3, x3)
        v4 = torch.mm(x4, x4)
        return torch.cat([v1, v2, v3, v4], 1)
# Inputs to the model
x1 = torch.full([5, 2], value=1, dtype=torch.uint8, requires_grad=True)
x2 = torch.full([5, 2], value=1, dtype=torch.uint8, requires_grad=True)
x3 = torch.full([5, 2], value=1, dtype=torch.uint8, requires_grad=True)
x4 = torch.full([5, 2], value=1, dtype=torch.uint8, requires_grad=True)

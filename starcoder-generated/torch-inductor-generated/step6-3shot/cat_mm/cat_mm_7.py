
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x1], 1)
        v1 = torch.transpose(v1, 0, 1)
        v1 = torch.mm(v1, x1)
        v1 = torch.mm(v1, x2)
        v2 = torch.cat([x1, x1], 1)
        v2 = torch.transpose(v2, 0, 1)
        v2 = torch.mm(v2, x1)
        v2 = torch.mm(v2, x2)
        v2 = torch.cat([v1, v1], 1)
        v2 = torch.transpose(v2, 0, 1)
        v2 = torch.mm(v2, x1)
        v2 = torch.mm(v2, x2)
        return v2
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)

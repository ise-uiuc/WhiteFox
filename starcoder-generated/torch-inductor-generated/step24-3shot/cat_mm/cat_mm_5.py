
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v10 = v1.repeat(2, 1) # Repeat along one dimension
        v2 = torch.mm(x1, x2)
        v20 = v2.repeat(2, 1)
        t1 = torch.cat([v1, v2], 1)
        t2 = torch.cat([v10, v20], 1)
        t3 = torch.cat([t1, t2], 0)
        t4 = torch.cat([t1, t2], 1)
        return torch.cat([t3, t4], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)

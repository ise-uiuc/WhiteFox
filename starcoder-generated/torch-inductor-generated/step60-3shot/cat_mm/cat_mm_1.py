
    
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.cat([v1, v1], 1)
        v3 = torch.mm(x1, x2)
        v4 = torch.cat([v3, v3], 1)
        v5 = torch.cat([v2, v4], 1) # v5: (3, 9)
        return torch.mm(x1, x2)
# Inputs to the model
x1 = torch.randn(5, 8)
x2 = torch.randn(8, 1)

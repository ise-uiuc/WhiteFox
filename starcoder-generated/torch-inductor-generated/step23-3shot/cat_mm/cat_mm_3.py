
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.randn(3, 2) # New inputs must be provided for all models
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(x2, x3)
        v4 = torch.mm(x1, x3)
        v5 = torch.mm(x1, x3)
        return torch.cat([v1, v2, v2, v3, v4, v5], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)

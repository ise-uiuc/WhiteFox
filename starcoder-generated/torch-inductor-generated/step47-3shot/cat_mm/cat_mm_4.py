
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x2, x2)
        v2 = torch.mm(x2, x2)
        return torch.cat([v1, v2], dim=-1)
# Inputs to the model
x1 = torch.rand(6, 1)
x2 = torch.rand(1, 6)

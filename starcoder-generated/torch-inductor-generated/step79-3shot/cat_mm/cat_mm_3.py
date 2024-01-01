
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        v = v.unsqueeze(0)
        v = torch.cat([v, v, v, v, v], 1)
        return v
# Inputs to the model
x1 = torch.randn(2, 1)
x2 = torch.randn(1, 2)

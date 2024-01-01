
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v = []
        v.append(torch.mm(x1, x1))
        v = torch.cat(v, 1)
        return torch.cat(x1, 1)
# Inputs to the model
x1 = torch.randn(5, 5)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v = []
        i = 0
        while i < 10:
            v.append(torch.mm(x1, x1))
            i += 1
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(5, 5)

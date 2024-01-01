
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        v.append(x1)
        for i in range(5, 10):
            v.append(i)
        v.append(x2)
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v = []
        for i in range (3):
            v.append(torch.mm(x, x))
        return torch.cat(v + v, 1)
# Inputs to the model
x = torch.randn(20, 5)

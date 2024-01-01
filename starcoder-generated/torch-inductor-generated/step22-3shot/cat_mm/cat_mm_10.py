
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        m1 = list()
        for i in range(10):
            v1 = torch.mm(x1, x2)
            m1.append(v1)
        return torch.cat(m1, 0)
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(3, 1)

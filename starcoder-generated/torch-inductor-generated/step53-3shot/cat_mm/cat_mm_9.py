
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        size = (2 - x2.size(1)) // 4
        if size > 0:
            v1 = torch.mm(x1, x2)
        else:
            v1 = torch.mm(x2.T, x1.T).T
        v1 = torch.cat([v1, v1, v1, v1], 1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(2, 2)

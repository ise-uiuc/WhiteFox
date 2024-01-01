
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.transpose(x, 0, 1)
        v2 = torch.mm(v1, v1)
        return v2 if (v1 > v2).all() else v1
# Inputs to the model
x = torch.randn(1, 5)
y = torch.randn(1, 5)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v = torch.mm(x1, x1)
        for i in range(10):
            v = torch.mm(v, v)
        return v
# Inputs to the model
x1 = torch.randn(5, 5)

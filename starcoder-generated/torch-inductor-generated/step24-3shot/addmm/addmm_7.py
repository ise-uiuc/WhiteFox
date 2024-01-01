
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        v = torch.mm(v, x2)
        return v
# Inputs to the model
x1 = torch.randn(3, 2)
x2 = torch.randn(2, 4)

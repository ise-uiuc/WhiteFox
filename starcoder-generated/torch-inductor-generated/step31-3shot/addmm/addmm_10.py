
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v3 = torch.mm(x2, x2)
        v2 = torch.mm(v3, v3)
        v1 = torch.mm(v2, v2)
        return v1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)

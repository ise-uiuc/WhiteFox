
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v0 = x1
        v1 = torch.mm(v0, v0) + x1
        return v1
# Inputs to the model
x1 = torch.randn(3, 3)
inp = torch.randn(3, 3)

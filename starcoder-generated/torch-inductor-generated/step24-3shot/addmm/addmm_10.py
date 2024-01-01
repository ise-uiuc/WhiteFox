
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.mm(x, y)
        v2 = torch.mul(v1, x)
        return v2
# Inputs to the model
x = torch.randn(1, 224)
y = torch.randn(224, 2)

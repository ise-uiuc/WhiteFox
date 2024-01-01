
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, other=0):
        v1, o1 = x1.chunk(2, dim=1)
        if other > 0:
            v2, o2 = x1.chunk(2, dim=2)
        v3 = v1 * float(other) + o1 + o2
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)

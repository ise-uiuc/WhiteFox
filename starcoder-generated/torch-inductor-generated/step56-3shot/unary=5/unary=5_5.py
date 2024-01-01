
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = 0.7071067811865476 * torch.erf(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 28, 3)

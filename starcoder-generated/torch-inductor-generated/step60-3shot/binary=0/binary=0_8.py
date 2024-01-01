
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, other=3.0):
        v1 = torch.atan(x1) + other
        return v1
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)

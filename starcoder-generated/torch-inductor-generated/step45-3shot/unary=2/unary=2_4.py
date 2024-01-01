
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.mul(x1, x1)
        return v1
# Inputs to the model
x1 = torch.randn(10, 20, 100, 10)

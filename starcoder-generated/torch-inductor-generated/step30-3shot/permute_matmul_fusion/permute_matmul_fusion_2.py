
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x3 = torch.mul(x1, x2)
        x4 = torch.mul(x1, x2)
        x5 = torch.mul(x3, x4)
        x6 = torch.mul(x5, x3)
        return (x5, x6)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)

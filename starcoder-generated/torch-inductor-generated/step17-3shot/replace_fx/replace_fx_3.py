
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x2 = x + 1
        x3 = torch.rand_like(x2)
        x4 = x2 + x3 + 1
        x5 = x3 + x4 + 1 + torch.rand_like(x4)
        x6 = x4 + x5 * x6 - 1
        x7 = x6 @ (x7[:, None] * x7[None, :]) - 1
        return x7
# Inputs to the model
x1 = torch.randn(10)

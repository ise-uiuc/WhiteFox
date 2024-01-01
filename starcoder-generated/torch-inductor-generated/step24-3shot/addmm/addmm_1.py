
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.matmul(x, x)
        t = x + x1
        return t
# Inputs to the model
x = torch.randn(12, 1, requires_grad=True)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        a = torch.rand_like(x)
        b = torch.rand_like(y)
        return 1
# Inputs to the model
x1 = [torch.randn(1, 2, 2), torch.randn(1, 2, 2)]
y1 = torch.randn(1, 2, 2)

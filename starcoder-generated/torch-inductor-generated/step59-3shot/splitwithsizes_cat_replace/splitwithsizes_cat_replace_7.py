
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = x
        b, c = torch.split(x, [1, 1, x.shape[-1] - 2], dim=-1)
        return (a-b+c, a, b, c)
# Inputs to the model
x1 = torch.randn(1, 30)

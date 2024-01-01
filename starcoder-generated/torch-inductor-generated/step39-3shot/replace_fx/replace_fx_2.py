
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t = torch.rand_like(x)
        return t
# Inputs to the model
x1 = torch.randn(1, 2)

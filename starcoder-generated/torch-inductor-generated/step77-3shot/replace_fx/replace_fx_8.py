
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.rand_like(x[0])
        t1 = torch.rand_like(x)
        return x + t1
# Inputs to the model
x = torch.randn(1, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.cat([x, x, x], -1)
        x = t1 + x
        return x
# Inputs to the model
x = torch.randn(5, 3, 4)

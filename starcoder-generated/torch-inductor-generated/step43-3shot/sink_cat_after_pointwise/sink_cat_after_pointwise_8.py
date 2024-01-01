
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = x + x
        b = a + a
        c = x + b
        return c
# Inputs to the model
x = torch.randn(5, 2)

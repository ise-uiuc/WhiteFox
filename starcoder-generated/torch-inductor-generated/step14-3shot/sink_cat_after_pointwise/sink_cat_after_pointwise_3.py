
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.cat([x, x], dim=1)
        t2 = torch.tanh(x)
        x = t1
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)

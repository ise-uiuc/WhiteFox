
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.clone()
        y[1., 1., 0.] = 2.
        return y
# Inputs to the model
x = torch.randn(2, 2, 2)

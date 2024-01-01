
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        y = y.tanh()
        return y
# Inputs to the model
x = torch.randn(1, 1, 2)

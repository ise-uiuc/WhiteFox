
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        y = y.tanh()
        x = torch.cat((y, y), dim=1).tanh()
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        y = torch.tanh(y)
        y = y.view(-1, 4)
        return y
# Inputs to the model
x = torch.randn(3, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x * 2
        y = torch.cat((y, y), dim=1).view(y.shape[0], -1)
        return y.tanh()
# Inputs to the model
x = torch.randn(5, 3, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x
        for i in range(5):
            y = torch.cat((x, x, x, x), dim=1)
            y = y.view(y.shape[0], -1)
        return x.view(x.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)

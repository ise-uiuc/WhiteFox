
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.nn.ReLU()(x)
        y = torch.cat((x, x), dim=1)
        if y.dim() == 2:
            y = y.tanh()
        else:
            y = y.view(-1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)

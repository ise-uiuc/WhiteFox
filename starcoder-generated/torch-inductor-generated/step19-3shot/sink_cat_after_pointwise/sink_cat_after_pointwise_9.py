
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        y = x.view(-1).tanh()
        return y.view(y.shape[0], -1)
# Inputs to the model
x = torch.randn(2, 3, 4)

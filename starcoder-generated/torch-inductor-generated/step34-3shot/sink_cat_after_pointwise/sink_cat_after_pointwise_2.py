
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        x = torch.cat((y, y.tanh()), dim=1)
        x = x.view(x.shape[0], -1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.arange(3)
        y = y.repeat(x.shape[0])
        y = y+1
        x = x.view(x.shape[0], -1)
        x = y.repeat(x.shape[1], 1)
        return torch.cat((y.unsqueeze(1), x, x), dim=1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 2)

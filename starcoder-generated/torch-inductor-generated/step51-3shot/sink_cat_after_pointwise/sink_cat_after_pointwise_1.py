
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x,x,x), -1)
        y = y.view(y.shape[0], -1)
        y = y.tanh()
        x = y.sum(dim=-1)
        return y
# Inputs to the model
x = torch.randn(2,3, 4)

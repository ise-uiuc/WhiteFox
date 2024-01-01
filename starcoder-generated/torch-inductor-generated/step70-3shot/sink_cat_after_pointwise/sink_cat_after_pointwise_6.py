
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        l = []
        l.append(x)
        l.append(x)
        x = torch.cat(l, dim=0)
        x = x.view(x.shape[0], -1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)

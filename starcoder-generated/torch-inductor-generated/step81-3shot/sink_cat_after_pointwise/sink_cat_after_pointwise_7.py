
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x
        x = torch.cat((x, x, x), dim=2).sum(dim=1)
        if x.shape == (4, 2):
            return x.tanh()
        if x.shape == (8, 2):
            return x.relu()
# Inputs to the model
x = torch.ones(2, 2, 2)

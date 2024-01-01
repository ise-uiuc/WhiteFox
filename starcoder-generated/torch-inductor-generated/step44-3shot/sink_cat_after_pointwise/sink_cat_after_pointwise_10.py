
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1) # concat three times
        y = x.view(x.shape[0], -1)
        y = y.tanh()
        if y.shape[0] == 1:
            y = y.relu()
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)

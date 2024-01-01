
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1).tanh()
        if y.shape[0] == 2:
            y = y.tanh()
        else:
            y = y.tanh()
        x = torch.cat((y, y), dim=0)
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)

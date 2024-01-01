
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y1 = y.view(y.shape[0], -1)
        y = y1.tanh()
        y = torch.cat((y, y), dim=0)
        y2 = y.view(y.shape[0], -1)
        if y2.shape[0] == 2:
            y = y.tanh()
        elif y2.shape[1] == 2:
            y = y.tanh()
        else:
            y = y.tanh()
        y = torch.relu(y)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)

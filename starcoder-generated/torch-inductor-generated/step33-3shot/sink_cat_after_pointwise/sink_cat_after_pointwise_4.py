
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y1 = torch.cat((y, y), 0)
        y2 = torch.tanh(y1)
        if x.shape[0] >= 2: y1 = torch.tanh(y1)
        return y2
# Inputs to the model
x = torch.rand((4, 5, 3))

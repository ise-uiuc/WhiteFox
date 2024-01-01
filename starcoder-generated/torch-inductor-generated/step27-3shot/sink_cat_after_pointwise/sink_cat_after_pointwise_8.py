
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = x.add(x)
        y2 = torch.cat((y1, x), dim=1).view(y1.shape[0], -1).tanh()
        y3 = torch.cat((y1, y2), dim=1).view(y1.shape[0], -1).tanh()
        return y3
# Inputs to the model
x = torch.randn(3, 3, 4)

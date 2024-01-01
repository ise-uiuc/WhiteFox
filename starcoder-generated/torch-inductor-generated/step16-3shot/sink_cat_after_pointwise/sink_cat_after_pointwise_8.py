
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        m = torch.nn.Conv2d(1, 30, 1, 1, bias=True)
        y1 = m(x)
        y2 = (y1 + y1).view(-1)
        y3 = (y1 + y2).view(-1)
        return torch.relu(y3)

# Inputs to the model
x = torch.randn(1, 1, 1, 1)

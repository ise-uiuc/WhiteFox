
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = torch.cat((x, x), dim=1)
        y2 = y1.view(-1).view(y1.shape[0], 2, -1)
        y3 = y2.tanh()
        return (y3, x)
# Inputs to the model
x = torch.randn(2, 3, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = x * 2
        y2 = y1 + 3
        y3 = torch.cat((y1, y2, y1), dim=1)
        y3 = y3.view(y3.shape[:-1])
        y3 = y3.tanh()
        return y3 + x
# Inputs to the model
x = torch.randn(5, 3, 4)

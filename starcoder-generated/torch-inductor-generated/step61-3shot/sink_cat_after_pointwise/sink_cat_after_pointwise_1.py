
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x), dim=0)
        y = y + x
        x = (y + y + y)[0]
        return x[0]
# Inputs to the model
x = torch.randn(2, 2)

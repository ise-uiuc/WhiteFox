
class SinkCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = (x, x, x).pop()
        y = (x, x, x).pop()
        z = y + 10
        x = z * 2 + 3
        x = x.tanh()
        return x
# Inputs to the model
x = torch.randn(2, requires_grad=True)

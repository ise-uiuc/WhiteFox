
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        z = x.view(-1)
        if y.shape[0] == z.shape[0]:
            z = z.sum()
        else:
            z = z.tanh()
        return z
# Inputs to the model
x = torch.randn(1, 2)
y = torch.randn(4, 2)

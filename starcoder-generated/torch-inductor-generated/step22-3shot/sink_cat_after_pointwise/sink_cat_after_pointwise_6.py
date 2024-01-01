
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.mean(dim=1)
        y = torch.cat((y, y), dim=1)
        x = y.tanh() if y.shape == (1, 2) else y.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)

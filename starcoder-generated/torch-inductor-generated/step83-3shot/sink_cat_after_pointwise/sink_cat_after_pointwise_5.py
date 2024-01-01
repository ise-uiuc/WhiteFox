
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=0)
        return y[None,...]
# Inputs to the model
x = torch.randn(2, 2, 2)

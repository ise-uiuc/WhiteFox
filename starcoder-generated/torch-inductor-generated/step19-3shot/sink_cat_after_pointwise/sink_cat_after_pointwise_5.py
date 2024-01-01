
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = torch.cat((y, y), dim=1)
        return y.view(-1)
# Inputs to the model
x = torch.randn(1, 2, 3)

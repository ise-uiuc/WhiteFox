
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x), dim=0)
        return y.view(y.shape[0], -1)
# Inputs to the model
x = torch.randn(2, 3, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x), dim=1)
        return y.view(1, y.shape[0], -1) if y.shape[1]!= 1 else.view(1, 2, -1) if y.shape[1]!= 2 else (1, 3, -1)
# Inputs to the model
x = torch.randn(1, 6, 4)

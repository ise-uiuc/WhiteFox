
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(1, -1)
        z = x.view(x.shape[0], -1)
        return y
# Inputs to the model
x = torch.randn(1, 2, 2)

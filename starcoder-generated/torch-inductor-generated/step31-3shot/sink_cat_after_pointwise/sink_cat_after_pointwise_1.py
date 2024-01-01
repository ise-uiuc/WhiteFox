
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        return y.view(y.shape[0], 1, -1) if y.shape!= (0, 0) else y
# Inputs to the model
x = torch.randn(2, 3, 4)

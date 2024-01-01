
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        if y.shape[0] == 1:
            y = y.reshape(-1)
        y = y.flatten()
        y = y.matmul(y.T)
        y = y.view(y.shape[0], -1)
        y = y.sigmoid()
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.transpose(0, 1)
        x = torch.cat(tensors=(x, x, x))
        y = x.view(3, 1, -1)
        z = y.transpose(0, 1).reshape(y.shape[0] * y.shape[1], -1)
        return z
# Inputs to the model
x = torch.randn(2)

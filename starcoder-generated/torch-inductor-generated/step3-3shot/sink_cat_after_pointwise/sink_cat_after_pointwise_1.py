
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        y = torch.cat((y, y), dim=1)
        x = y.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        return x + 1
# Inputs to the model
x = torch.randn(2, 3, 4)

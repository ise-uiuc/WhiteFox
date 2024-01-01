
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.reshape(x.shape[0], 10)
        z = torch.cat((y, y), dim=1)
        return z.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)

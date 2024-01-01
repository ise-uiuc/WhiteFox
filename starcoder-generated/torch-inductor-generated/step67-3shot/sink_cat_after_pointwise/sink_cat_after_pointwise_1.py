
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x0 = torch.cat([x, x], dim=1)
        x1 = x0.reshape(x.shape[0], -1).tanh()
        return x1
# Inputs to the model
x = torch.randn(3, 2)

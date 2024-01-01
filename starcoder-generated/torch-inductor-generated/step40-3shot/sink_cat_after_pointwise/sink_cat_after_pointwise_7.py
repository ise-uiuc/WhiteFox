
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([].append(x).append(x), dim=1)
        return y.view(y.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)

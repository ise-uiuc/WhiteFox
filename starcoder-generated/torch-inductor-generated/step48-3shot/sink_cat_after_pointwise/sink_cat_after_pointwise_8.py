
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(1, x.shape[0], -1).transpose(1, 2).view(x.shape[0], -1).transpose(0, 1)
        return torch.pow(x, 2.0).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)

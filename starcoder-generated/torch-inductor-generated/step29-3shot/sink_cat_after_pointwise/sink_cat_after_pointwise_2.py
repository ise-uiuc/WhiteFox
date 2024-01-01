
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(2, 4)
        return y.view(4).tanh()
# Inputs to the model
x = torch.randn(2, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.tanh(x)
        x2 = x1.view(x1.size(0), -1)
        x3 = x1.view(x1.size(0), -1)
        y = x3*(x1+x3)
        return y
# Inputs to the model
x = torch.randn(3, 4)

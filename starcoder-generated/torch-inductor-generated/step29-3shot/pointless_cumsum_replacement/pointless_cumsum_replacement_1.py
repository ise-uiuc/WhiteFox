
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = torch.mul(x, x)
        return a
# Inputs to the model
x = torch.randn(2)

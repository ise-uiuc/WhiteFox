
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, x1):
        v1 = torch.cat((x, x), dim=1)
        v2 = torch.cat((v1, x1), dim=1)
        y = torch.tanh(v2)
        return y
# Inputs to the model
x = torch.randn(1, 2)
x1 = torch.randn(1, 3)

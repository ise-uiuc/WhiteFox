
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat((x1, x1, x2, x2), dim=1)
        v2 = torch.cat((x1, x1, x2), dim=1)
        y = torch.tanh(v2)
        return y
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)

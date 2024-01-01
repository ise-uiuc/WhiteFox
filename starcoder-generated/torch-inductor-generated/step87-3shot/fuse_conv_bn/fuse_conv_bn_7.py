
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1 + x2
        v2 = torch.cat((x2, v1))
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
x2 = torch.randn(1, 2, 4, 4)

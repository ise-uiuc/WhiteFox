
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([torch.mm(x1, x2)], 0)
        return v1
# Inputs to the model
x1 = torch.randn(2,)
x2 = torch.randn(0,)

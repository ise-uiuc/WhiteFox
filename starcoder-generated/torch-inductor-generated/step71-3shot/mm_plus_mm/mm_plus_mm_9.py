
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x1, x2, x3):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x1) + torch.mm(x3, x2)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(7, 3)
x2 = torch.randn(7, 3)
x3 = torch.randn(3, 7)

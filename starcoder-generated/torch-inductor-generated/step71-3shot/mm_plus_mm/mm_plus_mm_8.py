
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x1, x2, x3):
        m1 = torch.mm(x1, x3)
        m2 = torch.mm(m1, x2)
        return m2
# Inputs to the model
x1 = torch.randn(7, 7)
x2 = torch.randn(7, 7)
x3 = torch.randn(7, 7)

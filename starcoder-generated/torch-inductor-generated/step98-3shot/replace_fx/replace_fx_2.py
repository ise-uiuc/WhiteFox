
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.randn(1)
        x4 = x3 + torch.rand_like(x3)
        x5 = torch.bmm(x1, x2)
        x6 = x4 + x5
        x7 = torch.addmm(x2, x4, x5)
        x8 = torch.bmm(x3, x7)
        return (x4, x6, x8)
# Inputs to the model
x1 = torch.randn(2, 2, 2, 2)
x2 = torch.randn(2, 2, 2, 2)

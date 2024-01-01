
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5):
        m1 = torch.mm(x1, x2)
        m2 = torch.mm(x3, x4)
        m3 = torch.mm(x5, x5)
        m4 = m1 + m2 + m3
        return m4
# Inputs to the model
x1 = torch.randn(5, 4)
x2 = torch.randn(4, 3)
x3 = torch.randn(5, 4)
x4 = torch.randn(4, 3)
x5 = torch.randn(4, 4)

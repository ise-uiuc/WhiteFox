
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        m1 = torch.mm(x1, x2)
        m2 = torch.mm(x2, x1)
        m4 = torch.mm(m1, m1)
        return m1 + m2 + m4
# Inputs to the model
x1 = torch.randn(7, 7)
x2 = torch.randn(7, 7)
x3 = torch.randn(7, 7)
x4 = torch.randn(7, 7)

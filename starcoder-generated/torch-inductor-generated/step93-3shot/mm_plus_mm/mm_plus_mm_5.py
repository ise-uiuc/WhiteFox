
class Model(torch.nn.Module):
    def forward(self, x):
        m1 = torch.mm(x, x)
        m2 = torch.mm(x, x)
        m3 = torch.mm(x, x)
        m4 = torch.mm(x, x)
        m5 = m1 + m2 + m3 + m4
        return m5
# Inputs to the model
x = torch.randn(9, 9)

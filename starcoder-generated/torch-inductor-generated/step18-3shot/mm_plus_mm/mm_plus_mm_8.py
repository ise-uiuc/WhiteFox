
class Model(torch.nn.Module):
    def forward(self, m1, m2, m3, m4):
        c1 = torch.mm(m1, m2)
        c2 = torch.mm(m3, m4)
        c3 = c1 + c2
        return c3
# Inputs to the model
m1 = torch.randn(3, 2)
m2 = torch.randn(3, 2)
m3 = torch.randn(3, 2)
m4 = torch.randn(3, 2)

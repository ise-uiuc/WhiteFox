
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        mm1 = torch.mm(x1, x2)
        mm2 = torch.mm(x1, x4)
        mm3 = torch.mm(x2, x3)
        mm4 = torch.mm(x4, x3)
        return mm1 + mm2 + mm3 + mm4
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)
x4 = torch.randn(2, 2)


class Model(torch.nn.Module):
    def forward(self, x1, x2):
        z1 = torch.mm(x1, x2)
        z2 = torch.mm(x2, x1)
        z3 = torch.mm(x1, x1)
        z4 = torch.mm(x2, x2)
        z5 = torch.mm(x2, x2)
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)

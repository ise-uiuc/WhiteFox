
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        a1 = torch.mm(x1, x1)
        a2 = torch.mm(x2, x2)
        b1 = torch.mm(x2, x1)
        b2 = torch.mm(x1, x2)
        return a1 + b1 + a2 + b2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)

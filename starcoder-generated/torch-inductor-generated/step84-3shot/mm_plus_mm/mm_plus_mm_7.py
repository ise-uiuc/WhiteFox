
class Model(nn.Module):
    def forward(self, x1, x2):
        a = torch.mm(x1, x2)
        b = torch.mm(a, a)
        c = torch.mm(x1, a)
        return b + c
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(7, 7)


class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        a1 = torch.mm(x1, x2)
        a1 = torch.mm(x2, x3)
        return a1 + a1
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
x3 = torch.randn(5, 5)

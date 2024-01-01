
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        a = torch.mm(x1, x2)
        b = torch.mm(x3, x4)
        c = torch.mm(x2, x1)
        x = a + b + c
        return x
# Inputs to the model
x1 = torch.randn(50, 100)
x2 = torch.randn(100, 50)
x3 = torch.randn(50, 50)
x4 = torch.randn(50, 100)

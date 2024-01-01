
class Model(torch.nn.Module):
    def forward(self, x0, x1, x2, x3):
        torch.mm(x0, x1)
        torch.mm(x2, x1)
        torch.mm(x1, x3)
        return torch.rand(1)
# Inputs to the model
x0 = torch.randn(3, 4)
x1 = torch.randn(4, 3)
x2 = torch.rand(2, 3)
x3 = torch.rand(3, 2)

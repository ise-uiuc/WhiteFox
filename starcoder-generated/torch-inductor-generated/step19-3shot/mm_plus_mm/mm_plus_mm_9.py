
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        v1 = torch.mm(x1, x3)
        v2 = torch.mm(x2, x3)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
x3 = torch.randn(5, 5)

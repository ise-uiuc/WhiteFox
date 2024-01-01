
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        v1 = torch.mm(x1, x2)
        return torch.sum(v1)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)

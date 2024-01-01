
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        o1 = torch.mm(x1, torch.transpose(x2, 1, 0))
        o2 = torch.mm(x3, torch.transpose(x4, 1, 0))
        return torch.mm(o1, o2)
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
x3 = torch.randn(5, 5)
x4 = torch.randn(5, 5)


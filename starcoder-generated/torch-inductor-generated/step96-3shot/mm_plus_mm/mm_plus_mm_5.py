
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        xx1 = torch.mm(x1, x2)
        xx2 = torch.mm(x3, x4)
        return torch.mm(xx1, xx2)
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)
x4 = torch.randn(2, 2)

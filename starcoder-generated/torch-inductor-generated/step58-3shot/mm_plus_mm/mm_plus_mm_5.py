
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5, x6):
        o1 = torch.mm(x1, x2)
        o2 = torch.mm(x3, x4)
        o3 = torch.mm(x5, x6)
        return o1 + o2 + o3
# Inputs to the model
x1 = torch.randn(8, 8)
x2 = torch.randn(8, 8)
x3 = torch.randn(8, 8)
x4 = torch.randn(8, 8)
x5 = torch.randn(8, 8)
x6 = torch.randn(8, 8)

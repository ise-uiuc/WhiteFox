
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        t1 = torch.add(x1, x2)
        t2 = torch.add(x1, x4)
        t3 = torch.add(x2, x3)
        t4 = torch.add(x2, x4)
        return t1 + t2
# Inputs to the model
x1 = torch.randn(16, 16)
x2 = torch.randn(16, 16)
x3 = torch.randn(16, 16)
x4 = torch.randn(16, 16)

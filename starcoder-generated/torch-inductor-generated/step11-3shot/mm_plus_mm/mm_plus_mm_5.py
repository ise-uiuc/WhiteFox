
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        t1 = torch.mm(x1 + x2, x1 + x2 + x3)
        t2 = torch.mm(x1, x2)
        t3 = torch.mm(t1, t2)
        return t3
# Inputs to the model
x1 = torch.randn(3, 3, dtype=torch.int32)
x2 = torch.randn(3, 3, dtype=torch.int32)
x3 = torch.randn(3, 3, dtype=torch.int32)

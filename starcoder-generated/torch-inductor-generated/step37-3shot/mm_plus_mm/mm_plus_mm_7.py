
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        t1 = x2 - x1
        t2 = x2 - x3
        t3 = t1 + t2
        t4 = t3 - x4
        t5 = t4 - x2
        t6 = t5 + t1
        return t6
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
x3 = torch.randn(1, 1)
x4 = torch.randn(1, 1)

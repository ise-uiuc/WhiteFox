
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = x1.permute(0, 2, 1)
        u1 = x2.permute(0, 2, 1)
        t2 = torch.matmul(t1, u1)
        return t2.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)

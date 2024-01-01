
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        u1 = x1.permute(0, 2, 1)
        u2 = x2.permute(0, 2, 1)
        u3 = torch.bmm(u1, u2)
        return u3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)

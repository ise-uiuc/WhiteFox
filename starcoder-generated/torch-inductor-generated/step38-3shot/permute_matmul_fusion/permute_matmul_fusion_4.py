
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        a = x1.permute(0, 2, 1)
        b = x2.permute(0, 2, 1)
        x1 = torch.bmm(a, x2)
        x2 = torch.bmm(b, x1)
        x3 = torch.bmm(x2, x2)
        x4 = torch.bmm(x1, x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)

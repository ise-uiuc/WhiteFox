
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1_permute = x2.permute(0, 2, 1)
        x2 = x1.permute(0, 2, 1)
        a = torch.bmm(x1_permute, x2)
        b = a[0][0]
        return b
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)

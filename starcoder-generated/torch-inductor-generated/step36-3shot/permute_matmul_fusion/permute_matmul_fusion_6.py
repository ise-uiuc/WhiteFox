
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        a1 = torch.exp(x1) + 1
        a2 = a1.permute(0, 2, 1)
        a3 = torch.bmm(torch.exp(x2), a2)
        a4 = a3[..., 1]
        a5 = a4.permute(0, 2, 1)
        a6 = a5[..., [0, 1]]
        return (a1, a4, a6)
# Inputs to the model
x1 = torch.randn(1, 1, 4)
x2 = torch.randn(2, 1, 4)

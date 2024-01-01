
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        temp7 = torch.bmm(x1.permute(0, 2, 1), x2)
        temp8 = torch.bmm(x1.permute(0, 2, 1), x2.permute(0, 2, 1))
        temp9 = torch.bmm(x1, x2.permute(0, 2, 1))
        return ((temp7, temp8, temp9))
# Inputs to the model
x1 = torch.randn(1, 1, 1)
x2 = torch.randn(1, 1, 1)

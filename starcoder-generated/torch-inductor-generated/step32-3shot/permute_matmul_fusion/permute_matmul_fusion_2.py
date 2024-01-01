
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        x3 = torch.matmul(x1, x2)
        x4 = x2.permute(0, 2, 1)
        x5 = torch.bmm(x3, x2)
        x6 = x2.permute(0, 2, 1)
        x7 = torch.matmul(x5, x3)
        x8 = x2.permute(0, 2, 1)
        x9 = torch.matmul(x5, x7)
        x10 = torch.cat((x6, x7), 1)
        return x10
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        out1 = torch.bmm(torch.matmul(x3.permute(0, 2, 1), x1), x2.permute(0, 2, 1))
        out2 = torch.bmm(out1, out1)
        out3 = torch.matmul(torch.bmm(out2, out1), x2.permute(0, 2, 1))
        return out2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)

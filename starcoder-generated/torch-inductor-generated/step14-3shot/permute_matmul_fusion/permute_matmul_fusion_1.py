
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x1, x2):
        v3 = torch.matmul(x1.permute(0, 2, 1), x2)
        return v3

x1 = torch.randn(25)
x2 = torch.randn(3, 5)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v10 = torch.bmm(x1.permute(0, 2, 1), x1.permute(0, 2, 1))
        v12 = torch.matmul(x1.permute(0, 2, 1), x1.permute(0, 2, 1))
        v14 = torch.matmul(x1.permute(0, 2, 1), x1.permute(0, 2, 1))
        return v10, v12, v14
# Inputs to the model
x1 = torch.randn(2, 2, 2)
x2 = torch.randn(2, 2, 2)

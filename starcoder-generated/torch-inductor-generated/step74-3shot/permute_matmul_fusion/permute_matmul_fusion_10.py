
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.r = torch.nn.ReLU()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = torch.matmul(v1, v2)
        v4 = torch.matmul(x1, x2)
        v5 = torch.matmul(x1, v2)
        v5[0][0] = 1.0
        v6 = torch.matmul(v1, x2)
        v7 = torch.matmul(x1, v2)
        v7[0][0] = 5.0
        return self.r(v3 + v4 + v5 + v6 + v7)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)

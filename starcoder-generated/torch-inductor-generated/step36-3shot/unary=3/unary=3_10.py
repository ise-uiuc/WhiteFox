
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.transpose(v1, 1, 1)
        v3 = torch.inverse(v2)
        v4 = torch.triangular_solve(v3.contiguous(), torch.cholesky(v1.contiguous()))
        v5 = v4.solution
        v6 = torch.cholesky(v1.contiguous())
        v7 = torch.transpose(v1, 1, 1)
        v8 = torch.inverse(v2)
        v9 = torch.triangular_solve(v8.contiguous(), v6)
        v10 = v9.solution
        v11 = v5 + 1
        v12 = v10 + 1
        v13 = v7 * v11
        v14 = v12 * v13
        v15 = torch.bmm(v14, v14)
        return v14
# Inputs to the model
x1 = torch.randn(1, 8, 112, 112)

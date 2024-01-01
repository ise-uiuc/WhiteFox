
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0, x2):
        v0 = x0.permute(0, 2, 1)
        v1 = x0.permute(0, 2, 1)
        v2 = x0.permute(0, 2, 1)
        v3 = x2.permute(0, 2, 1)
        v0_1 = torch.matmul(v0, v1)
        v3_1 = torch.matmul(v3, v0)
        v6 = torch.matmul(v2, v0_1)
        v9 = torch.matmul(v3_1, v1)
        v10 = torch.tanh(v9)
        v11 = torch.matmul(v3_1, v3)
        v12 = torch.tanh(v11)
        v13 = torch.tanh(v12)
        return v13
# Inputs to the model
x0 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)

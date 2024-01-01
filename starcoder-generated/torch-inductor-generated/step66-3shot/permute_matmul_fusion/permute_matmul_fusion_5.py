
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = x1.permute(2, 0, 1)[1:, 2:, 0]
        v2 = x2.permute(2, 0, 1)
        v3 = v2[0:1] * v2[2:3] * v2[5:6]
        return torch.matmul(t1, v3)
# Inputs to the model
x1 = torch.randn(3, 2, 3, 3, 1, 3, 1)
x2 = torch.randn(3, 2, 3, 3)

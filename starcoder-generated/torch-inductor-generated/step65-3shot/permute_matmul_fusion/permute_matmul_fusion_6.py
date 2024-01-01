
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x1.permute(0, 2, 1)
        v1 = x2.permute(0, 2, 1)
        v2 = torch.matmul(v0, v1)
        v3 = v2[0][0][0]
        return v3
# Inputs to the model
x1 = torch.rand(1, 2, 2)
x2 = torch.rand(1, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v2 = x1.permute(0, 2, 1)
        v3 = torch.matmul(v2, x2)
        return v3[0][0][0]
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)

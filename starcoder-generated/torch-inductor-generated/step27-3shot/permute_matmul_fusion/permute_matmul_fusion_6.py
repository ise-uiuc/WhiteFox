
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2):
        v1 = x2.permute(2, 1, 0)
        x = torch.randn((2, 2))
        v2 = torch.matmul(x, v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)

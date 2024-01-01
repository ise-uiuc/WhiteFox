
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 1, 2)
        v2 = torch.matmul(v1, x2)
        return v2
# Inputs to the model
x1 = torch.randn(2, 1, 1)
x2 = torch.randn(2, 1, 1)

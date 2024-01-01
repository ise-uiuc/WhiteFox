
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a):
        x1 = a.permute(0, 2, 1)
        v0 = torch.matmul(x1, x1)
        return v0
# Inputs to the model
x0 = torch.randn(3, 2, 1)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x2.reshape(2, 2)
        v1 = x1.reshape(2, 2)
        v2 = v1.reshape(2, 2)
        v3 = torch.matmul(v0, v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2)

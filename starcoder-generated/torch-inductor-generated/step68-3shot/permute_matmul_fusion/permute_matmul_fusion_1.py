
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.reshape(x1.permute(0, 2, 1), (1, 2, 2))
        v2 = torch.reshape(x2.permute(0, 2, 1), (1, 2, 2))
        return torch.matmul(v1, v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(1, 0, 2)
        v2 = v1.transpose(0, 1)
        v3 = v2.permute(2, 1, 0)
        return torch.matmul(v3, x2)
# Inputs to the model
x1 = torch.randn(2, 1, 2)
x2 = torch.randn(1, 2, 2)

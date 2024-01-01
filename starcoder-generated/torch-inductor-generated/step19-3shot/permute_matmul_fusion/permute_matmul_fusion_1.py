
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.permute1 = torch.Tensor.permute
    def forward(self, x1, x2):
        v1 = self.permute1(x1, 0, 2, 1)
        v2 = self.permute1(x2, 0, 2, 1)
        v3 = torch.matmul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 3, 3)
x2 = torch.randn(2, 3, 3, 3)

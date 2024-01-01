
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2): # Swap last two dimensions
        v0 = x2.contiguous().permute(0, 1, 3, 2)
        return torch.matmul(x1, v0)
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
x2 = torch.randn(1, 1, 6, 4)

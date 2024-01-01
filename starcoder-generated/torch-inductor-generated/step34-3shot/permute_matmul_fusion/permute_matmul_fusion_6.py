
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.matmul(x1.permute(0, 2, 1), x2.permute(0, 2, 1)).permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2).to(torch.float64)
x2 = torch.randn(1, 2, 2).to(torch.float64)

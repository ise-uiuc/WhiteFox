
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return torch.matmul(x1, x1.permute(0, 2, 1).permute(0, 2, 1))
# Inputs to the model
x1 = torch.randn(7, 11, 2)

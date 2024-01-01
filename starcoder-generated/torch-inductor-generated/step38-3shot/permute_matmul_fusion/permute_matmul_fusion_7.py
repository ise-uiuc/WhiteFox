
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.matmul(torch.matmul(x1, x2), torch.matmul(x2, x1))
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)

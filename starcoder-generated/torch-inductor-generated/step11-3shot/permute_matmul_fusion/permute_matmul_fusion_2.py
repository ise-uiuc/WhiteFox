
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.bmm(x1.permute(0, 2, 1), x2.permute(2, 1, 0)).permute(0, 1, 3, 2)
# Inputs to the model
x1 = torch.randn(1, 2, 5, 2)
x2 = torch.randn(1, 2, 2, 3)

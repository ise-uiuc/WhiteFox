
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x2.permute(0, 2, 1)
        return torch.bmm(x1, v1)
# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 3, 2)

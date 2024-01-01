
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(2, 1, 0)
        return torch.bmm(v1, x2)
# Inputs to the model
x1 = torch.randn(2, 2, 1)
x2 = torch.randn(2, 2, 1)
